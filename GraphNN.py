import pandas as pd
import yfinance as yf
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tabulate import tabulate

# Parameters
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL', 'INTC', 'CSCO',
    'IBM', 'TXN', 'QCOM', 'NVDA', 'ADBE', 'CRM', 'AMD', 'AVGO',
    'HPQ', 'MU', 'MSI', 'ADP', 'EA', 'EBAY', 'INTU',
    'LRCX', 'MCHP', 'NTAP', 'PAYX', 'SNPS', 'STX', 'SWKS'
]
start_date = '2018-01-01'
end_date = '2020-12-31'
hidden_dim = 32
num_heads = 4
epochs = 20
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

returns = combined_data.pct_change().dropna()
correlation_matrix = returns.corr()

# Edge indices for different strategies
edge_index_arbitrage = create_edge_index(correlation_matrix, threshold=0.6)
edge_index_longshort = create_edge_index(correlation_matrix, threshold=0.75)

# Generate features and labels
def generate_features_labels(combined_data):
    returns = combined_data.pct_change().fillna(0)
    features = returns.shift(1).fillna(0)
    labels = returns
    return features, labels

features, labels = generate_features_labels(combined_data)
dates = features.index

# Create data objects for arbitrate and long-short strategies
def create_data_objects(features, labels, edge_index_arbitrage, edge_index_longshort):
    data_list_arbitrage = []
    data_list_longshort = []
    num_nodes = features.shape[1]
    for idx in range(len(features)):
        x = torch.tensor(features.iloc[idx].values, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(labels.iloc[idx].values, dtype=torch.float)

        data_arbitrage = Data(x=x, edge_index=edge_index_arbitrage, y=y)
        data_arbitrage.date = dates[idx]
        data_arbitrage.num_nodes = num_nodes
        data_list_arbitrage.append(data_arbitrage)

        data_longshort = Data(x=x, edge_index=edge_index_longshort, y=y)
        data_longshort.date = dates[idx]
        data_longshort.num_nodes = num_nodes
        data_list_longshort.append(data_longshort)
    return data_list_arbitrage, data_list_longshort

data_list_arbitrage, data_list_longshort = create_data_objects(features, labels, edge_index_arbitrage, edge_index_longshort)

# For momentum strategy, we can use the same data list
data_list_momentum = data_list_arbitrage

# Split into training and tests sets
split_idx = int(len(data_list_arbitrage) * 0.7)
train_data_arbitrage = data_list_arbitrage[:split_idx]
test_data_arbitrage = data_list_arbitrage[split_idx:]

train_data_momentum = data_list_momentum[:split_idx]
test_data_momentum = data_list_momentum[split_idx:]

train_data_longshort = data_list_longshort[:split_idx]
test_data_longshort = data_list_longshort[split_idx:]

# Benchmark against S&P 500 and Technology Index Fund
benchmark = yf.download('^GSPC', start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()
benchmark = benchmark[dates[0]:dates[-1]]
tech_index = yf.download('QQQ', start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()
tech_index = tech_index[dates[0]:dates[-1]]

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
        if strategy_name == 'Momentum Trading on Diffusion Spikes':
            iter_range = range(seq_length, len(train_data))
            train_iter = tqdm(iter_range, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
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
                avg_loss = total_loss / (idx - seq_length + 1)
                train_iter.set_postfix({'Loss': avg_loss})
        else:
            train_iter = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for data in train_iter:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / (train_data.index(data)+1)
                train_iter.set_postfix({'Loss': avg_loss})
    return model

# Backtest on historical data
def backtest(model, test_data, device, strategy_name):
    model = model.to(device)
    model.eval()
    predictions = []
    actuals = []
    dates = []
    seq_length = 5  # Sequence length for LSTM

    with torch.no_grad():
        if strategy_name == 'Momentum Trading on Diffusion Spikes':
            iter_range = range(seq_length, len(test_data))
            test_iter = tqdm(iter_range, desc=f"Backtesting {strategy_name}", leave=False)
            for idx in test_iter:
                data_seq = test_data[idx - seq_length:idx]
                x_seq_list = [d.x.to(device) for d in data_seq]
                edge_index_list = [d.edge_index.to(device) for d in data_seq]
                y = test_data[idx].y.mean().to(device)
                date = test_data[idx].date

                out = model(x_seq_list, edge_index_list)
                predictions.append(out.cpu().numpy())
                actuals.append(y.cpu().numpy())
                dates.append(date)
        else:
            test_iter = tqdm(test_data, desc=f"Backtesting {strategy_name}", leave=False)
            for data in test_iter:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                predictions.extend(out.cpu().numpy())
                actuals.extend(data.y.cpu().numpy())
                dates.extend([data.date] * data.num_nodes)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    dates = pd.to_datetime(dates)
    return predictions, actuals, dates

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

# Initialize and train models
strategy_models = {}
strategy_results = {}

# Strategy 1: arbitrage on lagged price movements
print("\nTraining: Arbitrage on Lagged Price Movements")
arbitrage_model = ArbitrageModel(num_features=1, hidden_dim=hidden_dim, num_heads=num_heads)
arbitrage_model = train(arbitrage_model, train_data_arbitrage, epochs, learning_rate, device, 'Arbitrage on Lagged Price Movements')
predictions_a, actuals_a, dates_a = backtest(arbitrage_model, test_data_arbitrage, device, 'Arbitrage on Lagged Price Movements')
cumulative_returns_a, benchmark_cum_returns_a, metrics_a, strategy_returns_a = evaluate_strategy(predictions_a, actuals_a, dates_a, benchmark, 'Arbitrage on Lagged Price Movements')
strategy_models['Arbitrage on Lagged Price Movements'] = arbitrage_model
strategy_results['Arbitrage on Lagged Price Movements'] = (cumulative_returns_a, metrics_a)

# Strategy 2: momentum trading on diffusion spikes
print("\nTraining: Momentum Trading on Diffusion Spikes")
momentum_model = MomentumModel(num_features=1, hidden_dim=hidden_dim, lstm_hidden_dim=hidden_dim)
momentum_model = train(momentum_model, train_data_momentum, epochs, learning_rate, device, 'Momentum Trading on Diffusion Spikes')
predictions_b, actuals_b, dates_b = backtest(momentum_model, test_data_momentum, device, 'Momentum Trading on Diffusion Spikes')
cumulative_returns_b, benchmark_cum_returns_b, metrics_b, strategy_returns_b = evaluate_strategy(predictions_b, actuals_b, dates_b, benchmark, 'Momentum Trading on Diffusion Spikes')
strategy_models['Momentum Trading on Diffusion Spikes'] = momentum_model
strategy_results['Momentum Trading on Diffusion Spikes'] = (cumulative_returns_b, metrics_b)

# Strategy 3: long-short basket trading on correlated stocks
print("\nTraining: Long-Short Basket Trading on Correlated Stocks")
long_short_model = LongShortModel(num_features=1, hidden_dim=hidden_dim, num_heads=num_heads)
long_short_model = train(long_short_model, train_data_longshort, epochs, learning_rate, device, 'Long-Short Basket Trading on Correlated Stocks')
predictions_c, actuals_c, dates_c = backtest(long_short_model, test_data_longshort, device, 'Long-Short Basket Trading on Correlated Stocks')
cumulative_returns_c, benchmark_cum_returns_c, metrics_c, strategy_returns_c = evaluate_strategy(predictions_c, actuals_c, dates_c, benchmark, 'Long-Short Basket Trading on Correlated Stocks')
strategy_models['Long-Short Basket Trading on Correlated Stocks'] = long_short_model
strategy_results['Long-Short Basket Trading on Correlated Stocks'] = (cumulative_returns_c, metrics_c)

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns_a.index, cumulative_returns_a['Returns'], label='Arbitrage on Lagged Price Movements')
plt.plot(cumulative_returns_b.index, cumulative_returns_b['Returns'], label='Momentum Trading on Diffusion Spikes')
plt.plot(cumulative_returns_c.index, cumulative_returns_c['Returns'], label='Long-Short Basket Trading on Correlated Stocks')
plt.plot(benchmark_cum_returns_a.index, benchmark_cum_returns_a.values, label='Benchmark (S&P 500)')
tech_index = tech_index[cumulative_returns_a.index[0]:cumulative_returns_a.index[-1]]
tech_cum_returns = (1 + tech_index).cumprod() - 1
plt.plot(tech_cum_returns.index, tech_cum_returns.values, label='Technology Index Fund (QQQ)')
plt.legend()
plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()

# Print metrics as table
metrics_data = [
    [metrics_a['Strategy'], metrics_a['Alpha'], metrics_a['Beta'], metrics_a['Sharpe Ratio']],
    [metrics_b['Strategy'], metrics_b['Alpha'], metrics_b['Beta'], metrics_b['Sharpe Ratio']],
    [metrics_c['Strategy'], metrics_c['Alpha'], metrics_c['Beta'], metrics_c['Sharpe Ratio']]
]

headers = ['Strategy', 'Alpha', 'Beta', 'Sharpe Ratio']
print(tabulate(metrics_data, headers=headers, tablefmt='grid'))