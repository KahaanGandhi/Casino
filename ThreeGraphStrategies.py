import pandas as pd
import yfinance as yf
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')

# Parameters
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL', 'INTC', 'CSCO',
    'IBM', 'TXN', 'QCOM', 'NVDA', 'ADBE', 'CRM', 'AMD', 'AVGO',
    'HPQ', 'MU', 'MSI', 'ADP', 'EA', 'EBAY', 'INTU',
    'LRCX', 'MCHP', 'NTAP', 'PAYX', 'SNPS', 'STX', 'SWKS'
]
start_date = '2018-01-01'
end_date = '2021-12-31'
hidden_dim = 32
num_heads = 4
epochs = 10
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
def load_stock_data(tickers, start_date, end_date):
    data = {}
    valid_tickers = []
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"No data for {ticker}. Skipping.")
                continue
            data[ticker] = df.copy()
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return data, valid_tickers

data, valid_tickers = load_stock_data(tickers, start_date, end_date)
if not valid_tickers:
    raise Exception("No valid tickers. Exiting.")

# Preprocess data
def preprocess_data(data, valid_tickers):
    combined = pd.DataFrame()
    for ticker in valid_tickers:
        df = data[ticker][['Adj Close']].copy()
        df = df.rename(columns={'Adj Close': ticker})
        combined = pd.concat([combined, df], axis=1)
    combined.fillna(method='ffill', inplace=True)
    combined.dropna(inplace=True)
    return combined

combined_data = preprocess_data(data, valid_tickers)

# Generate features and labels
def generate_features_labels(combined_data):
    returns = combined_data.pct_change().fillna(0)
    features = returns.shift(1).fillna(0)
    labels = returns
    return features, labels

features, labels = generate_features_labels(combined_data)
dates = features.index

# Create data objects without edge_index (will be assigned during backtesting)
def create_data_objects(features, labels):
    data_list = []
    num_nodes = features.shape[1]
    for idx in range(len(features)):
        x = torch.tensor(features.iloc[idx].values, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(labels.iloc[idx].values, dtype=torch.float)
        data_obj = Data(x=x, y=y)
        data_obj.date = features.index[idx]
        data_obj.num_nodes = num_nodes
        data_list.append(data_obj)
    return data_list

data_list = create_data_objects(features, labels)

# Benchmark against S&P 500 and Technology Index Fund
benchmark = yf.download('^GSPC', start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()
benchmark = benchmark[dates[0]:dates[-1]]
tech_index = yf.download('QQQ', start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()
tech_index = tech_index[dates[0]:dates[-1]]

# Create correlation matrix and edge indices
def create_edge_index(correlation_matrix, threshold):
    edge_index = []
    num_nodes = correlation_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and abs(correlation_matrix.iloc[i, j]) >= threshold:
                edge_index.append([i, j])
    if not edge_index:
        raise ValueError(f"No edges found with correlation threshold {threshold}. Adjust the threshold.")
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

# Models for Each Strategy
# Arbitrage on lagged price movements
class ArbitrageModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads):
        super(ArbitrageModel, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.functional.elu(x)
        x = self.conv2(x, edge_index)
        x = nn.functional.elu(x)
        x = self.fc(x)
        return x.squeeze()

# Momentum trading on diffusion spikes
class MomentumModel(nn.Module):
    def __init__(self, num_features, hidden_dim, lstm_hidden_dim):
        super(MomentumModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x_seq_list, edge_index_list):
        conv_outputs = []
        for x, edge_index in zip(x_seq_list, edge_index_list):
            x = self.conv1(x, edge_index)
            x = nn.functional.relu(x)
            conv_outputs.append(x)
        x_seq = torch.stack(conv_outputs, dim=0)
        x_seq = x_seq.mean(dim=1)
        x_seq = x_seq.unsqueeze(0)
        lstm_out, _ = self.lstm(x_seq)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

# Long-short basket trading on correlated stocks
class LongShortModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads):
        super(LongShortModel, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
        self.fc = nn.Linear(hidden_dim * num_heads, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = nn.functional.leaky_relu(x)
        x = self.fc(x)
        return x.squeeze()

# Training loop
def train(model, train_data, epochs, learning_rate, device, strategy_name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()
    seq_length = 5  # Sequence length for LSTM

    for epoch in range(epochs):
        total_loss = 0
        if strategy_name == 'Momentum Trading on Diffusion Spikes' or strategy_name == 'Momentum':
            iter_range = range(seq_length, len(train_data))
            train_iter = iter_range  # Remove tqdm here to simplify progress bars
            for idx in train_iter:
                data_seq = train_data[idx - seq_length:idx]
                x_seq_list = [d.x.to(device) for d in data_seq]
                edge_index_list = [d.edge_index.to(device) for d in data_seq]
                y = train_data[idx].y.mean().to(device)

                optimizer.zero_grad()
                out = model(x_seq_list, edge_index_list)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            train_iter = train_data
            for idx, data in enumerate(train_iter):
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    return model

# Backtest with walk-forward validation
def walk_forward_backtest(model_class, data_list, device, strategy_name):
    predictions = []
    actuals = []
    dates_list = []
    retrain_interval = 14  # Retrain the model every 14 days
    initial_train_size = 252  # Use the first 252 days (~1 year) for initial training

    # print(f"\nStarting walk-forward backtesting for {strategy_name}...")

    total_periods = (len(data_list) - initial_train_size) // retrain_interval
    backtest_progress = tqdm(range(initial_train_size, len(data_list) - 1, retrain_interval), desc=f"Backtesting {strategy_name}")

    for i in backtest_progress:
        # Define training and test sets
        train_data = data_list[:i]
        test_data = data_list[i:i + retrain_interval]

        if len(test_data) == 0:
            break

        # Prepare edge indices using training data only
        train_features = features.iloc[:i]
        train_labels = labels.iloc[:i]
        train_returns = train_labels
        correlation_matrix = train_returns.corr()

        # Create edge indices based on strategy
        if strategy_name == 'Arbitrage on Lagged Price Movements' or strategy_name == 'Momentum Trading on Diffusion Spikes' or strategy_name == "Statistical Arbitrage" or strategy_name == "Momentum":
            edge_index = create_edge_index(correlation_matrix, threshold=0.6)
        else:
            edge_index = create_edge_index(correlation_matrix, threshold=0.75)

        # Update data objects with new edge_index
        for data_obj in train_data + test_data:
            data_obj.edge_index = edge_index

        # Initialize and train the model
        if strategy_name == 'Momentum Trading on Diffusion Spikes' or strategy_name == 'Momentum':
            model = model_class(num_features=1, hidden_dim=hidden_dim, lstm_hidden_dim=hidden_dim)
        else:
            model = model_class(num_features=1, hidden_dim=hidden_dim, num_heads=num_heads)
        model = train(model, train_data, epochs, learning_rate, device, strategy_name)

        # Make predictions on test data
        model.eval()
        seq_length = 5  # For momentum strategy

        with torch.no_grad():
            if strategy_name == 'Momentum Trading on Diffusion Spikes' or strategy_name == 'Momentum':
                iter_range = range(seq_length, len(test_data))
                for idx in iter_range:
                    data_seq = test_data[idx - seq_length:idx]
                    x_seq_list = [d.x.to(device) for d in data_seq]
                    edge_index_list = [d.edge_index.to(device) for d in data_seq]
                    y = test_data[idx].y.mean().to(device)
                    date = test_data[idx].date

                    out = model(x_seq_list, edge_index_list)
                    predictions.append(out.cpu().numpy())
                    actuals.append(y.cpu().numpy())
                    dates_list.append(date)
            else:
                for data in test_data:
                    data = data.to(device)
                    out = model(data.x, data.edge_index)
                    predictions.extend(out.cpu().numpy())
                    actuals.extend(data.y.cpu().numpy())
                    dates_list.extend([data.date] * data.num_nodes)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    dates_list = pd.to_datetime(dates_list)
    return predictions, actuals, dates_list

# Calculate cumulative returns and metrics
def evaluate_strategy(predictions, actuals, dates, benchmark_returns, strategy_name):
    signals = np.sign(predictions)
    strategy_returns = signals * actuals
    strategy_returns = pd.DataFrame({'Returns': strategy_returns}, index=dates)
    # Group by date to get average return per day
    strategy_returns = strategy_returns.groupby(strategy_returns.index).mean()
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    sharpe_ratio = (strategy_returns['Returns'].mean() / strategy_returns['Returns'].std()) * np.sqrt(252)

    # Align benchmark returns
    benchmark_returns = benchmark_returns[strategy_returns.index[0]:strategy_returns.index[-1]]
    benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1

    # Calculate Alpha and Beta
    aligned_benchmark_returns = benchmark_returns.loc[strategy_returns.index]
    cov_matrix = np.cov(strategy_returns['Returns'], aligned_benchmark_returns)
    beta = cov_matrix[0,1] / cov_matrix[1,1]
    alpha = (strategy_returns['Returns'].mean() - beta * aligned_benchmark_returns.mean()) * 252

    metrics = {
        'Strategy': strategy_name,
        'Alpha': alpha,
        'Beta': beta,
        'Sharpe Ratio': sharpe_ratio
    }

    return cumulative_returns, benchmark_cum_returns, metrics, strategy_returns

# Initialize and backtest models
strategy_results = {}

# Strategy 1: arbitrage on lagged price movements
predictions_a, actuals_a, dates_a = walk_forward_backtest(ArbitrageModel, data_list, device, 'Statistical Arbitrage')
cumulative_returns_a, benchmark_cum_returns_a, metrics_a, strategy_returns_a = evaluate_strategy(predictions_a, actuals_a, dates_a, benchmark, 'Arbitrage on Lagged Price Movements')
strategy_results['Arbitrage on Lagged Price Movements'] = (cumulative_returns_a, metrics_a)

# Strategy 2: momentum trading on diffusion spikes
predictions_b, actuals_b, dates_b = walk_forward_backtest(MomentumModel, data_list, device, 'Momentum')
cumulative_returns_b, benchmark_cum_returns_b, metrics_b, strategy_returns_b = evaluate_strategy(predictions_b, actuals_b, dates_b, benchmark, 'Momentum Trading on Diffusion Spikes')
strategy_results['Momentum Trading on Diffusion Spikes'] = (cumulative_returns_b, metrics_b)

# Strategy 3: long-short basket trading on correlated stocks
predictions_c, actuals_c, dates_c = walk_forward_backtest(LongShortModel, data_list, device, 'Market Neutral')
cumulative_returns_c, benchmark_cum_returns_c, metrics_c, strategy_returns_c = evaluate_strategy(predictions_c, actuals_c, dates_c, benchmark, 'Long-Short Basket Trading on Correlated Stocks')
strategy_results['Long-Short Basket Trading on Correlated Stocks'] = (cumulative_returns_c, metrics_c)

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns_a.index, cumulative_returns_a['Returns'], label='Statistical Arbitrage')
plt.plot(cumulative_returns_b.index, cumulative_returns_b['Returns'], label='Momentum')
plt.plot(cumulative_returns_c.index, cumulative_returns_c['Returns'], label='Market Neutral')
plt.plot(benchmark_cum_returns_a.index, benchmark_cum_returns_a.values, label='Benchmark (S&P 500)')
tech_index = tech_index[cumulative_returns_a.index[0]:cumulative_returns_a.index[-1]]
tech_cum_returns = (1 + tech_index).cumprod() - 1
plt.plot(tech_cum_returns.index, tech_cum_returns.values, label='Technology Index Fund (QQQ)')
plt.legend()
plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')

# Print metrics as table
metrics_data = [
    [metrics_a['Strategy'], metrics_a['Alpha'], metrics_a['Beta'], metrics_a['Sharpe Ratio']],
    [metrics_b['Strategy'], metrics_b['Alpha'], metrics_b['Beta'], metrics_b['Sharpe Ratio']],
    [metrics_c['Strategy'], metrics_c['Alpha'], metrics_c['Beta'], metrics_c['Sharpe Ratio']]
]

headers = ['Strategy', 'Alpha', 'Beta', 'Sharpe Ratio']
print(tabulate(metrics_data, headers=headers, tablefmt='grid'))
plt.savefig('returns.png', dpi=800)
plt.show()
