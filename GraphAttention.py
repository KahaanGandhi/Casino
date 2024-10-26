import pandas as pd
import yfinance as yf
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

#========================
# Define hyperparameters
#========================

PARAMS = {
    "tickers": [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL', 'INTC', 'CSCO',
        'IBM', 'TXN', 'QCOM', 'NVDA', 'ADBE', 'CRM', 'AMD', 'AVGO',
        'HPQ', 'MU', 'MSI', 'ADP', 'EA', 'EBAY', 'INTU',
        'LRCX', 'MCHP', 'NTAP', 'PAYX', 'SNPS', 'STX', 'SWKS',
        'JNJ', 'JPM', 'WMT', 'PG', 'BAC', 'XOM', 'V', 'T', 'CVX', 'HD'
    ],
    "start_date": '2010-01-01',
    "end_date": '2020-12-31',
    "hidden_dim": 128,
    "num_heads": 8,
    "epochs": 10,
    "learning_rate": 0.0005,
    "retrain_interval": 21,         # Retrain every 21 days
    "initial_train_size": 252,      # Use the first 252 days for initial training
    "correlation_percentile": 80,   # Percentile for edge creation
    "window_rsi": 14,               # Window size for RSI calculation
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

print(f"Using device: {PARAMS['device']}")

#==========================
# Load and preprocess data
#==========================

# Load stock data from Yahoo Finance
def load_stock_data(tickers, start_date, end_date):
    data = {}
    valid_tickers = []
    # Loop over each ticker to download data
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"No data for {ticker}. Skipping.")
                continue
            df['Ticker'] = ticker
            data[ticker] = df.copy()
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return data, valid_tickers

# Load the data
data, valid_tickers = load_stock_data(PARAMS['tickers'], PARAMS['start_date'], PARAMS['end_date'])
if not valid_tickers:
    raise Exception("No valid tickers. Exiting.")

# Preprocess data and align dates
def preprocess_data(data, valid_tickers):
    combined_price = pd.DataFrame()
    combined_volume = pd.DataFrame()
    # Combine adjusted close prices and volumes into separate DataFrames
    for ticker in valid_tickers:
        df = data[ticker][['Adj Close', 'Volume']].copy()
        df = df.rename(columns={'Adj Close': ticker, 'Volume': f"{ticker}_Volume"})
        combined_price = pd.concat([combined_price, df[[ticker]]], axis=1)
        combined_volume = pd.concat([combined_volume, df[[f"{ticker}_Volume"]]], axis=1)
    # Forward-fill missing data and drop rows with any remaining NaNs
    combined_price.fillna(method='ffill', inplace=True)
    combined_price.dropna(inplace=True)
    combined_volume.fillna(0, inplace=True)
    # Align volume data with price data
    combined_volume = combined_volume.reindex(combined_price.index)
    return combined_price, combined_volume

# Preprocess the data
combined_price, combined_volume = preprocess_data(data, valid_tickers)
if combined_price.empty:
    raise Exception("Combined data is empty after preprocessing. Exiting.")

#=====================
# Feature engineering
#=====================

# Function to generate features and labels
def generate_features_labels(combined_price, combined_volume):
    # Calculate daily returns
    returns = combined_price.pct_change().fillna(0)
    # Shift returns to use as features (lagged returns)
    features = returns.shift(1).fillna(0)
    window = PARAMS['window_rsi']
    # Loop over each ticker to calculate technical indicators
    for ticker in valid_tickers:
        price = combined_price[ticker]
        # Calculate RSI
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features[f'{ticker}_RSI'] = rsi.fillna(0)
        # Calculate Moving Averages
        features[f'{ticker}_MA20'] = price.rolling(window=20).mean().fillna(0)
        features[f'{ticker}_MA50'] = price.rolling(window=50).mean().fillna(0)
        # Calculate Bollinger Bands
        std_dev = price.rolling(window=20).std().fillna(0)
        features[f'{ticker}_BB_upper'] = features[f'{ticker}_MA20'] + (std_dev * 2)
        features[f'{ticker}_BB_lower'] = features[f'{ticker}_MA20'] - (std_dev * 2)
    # Add shifted volume as a feature
    features = pd.concat([features, combined_volume.shift(1).fillna(0)], axis=1)
    # Labels are the current day's returns
    labels = returns
    return features, labels

# Generate features and labels
features, labels = generate_features_labels(combined_price, combined_volume)
dates = features.index

# Normalize features using StandardScaler
scaler = StandardScaler()
features[features.columns] = scaler.fit_transform(features)

#================================
# Construct Data Objects for GNN
#================================

# Function to create data objects for each day
def create_data_objects(features, labels):
    data_list = []
    num_nodes = len(valid_tickers)
    # Determine number of features per node
    num_node_features = 8  # Adjust based on the number of features per node
    for idx in range(len(features)):
        x = []
        # Extract features for each ticker
        for ticker in valid_tickers:
            feature_vector = features.iloc[idx][[
                ticker,
                f'{ticker}_RSI',
                f'{ticker}_MA20',
                f'{ticker}_MA50',
                f'{ticker}_BB_upper',
                f'{ticker}_BB_lower',
                f"{ticker}_Volume"
            ]].values
            x.append(feature_vector)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(labels.iloc[idx].values, dtype=torch.float)
        data_obj = Data(x=x, y=y)
        data_obj.date = features.index[idx]
        data_obj.num_nodes = num_nodes
        data_list.append(data_obj)
    return data_list

# Create data objects
data_list = create_data_objects(features, labels)

#=================================
# Benchmarks to evaluate strategy
#=================================

# Download benchmark data (S&P 500 and Technology Index Fund)
benchmark = yf.download('^GSPC', start=PARAMS['start_date'], end=PARAMS['end_date'], progress=False)['Adj Close'].pct_change().dropna()
benchmark = benchmark[dates[0]:dates[-1]]
tech_index = yf.download('QQQ', start=PARAMS['start_date'], end=PARAMS['end_date'], progress=False)['Adj Close'].pct_change().dropna()
tech_index = tech_index[dates[0]:dates[-1]]

#==================
# Define GNN Model
#==================

class AdvancedGNNModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads):
        super(AdvancedGNNModel, self).__init__()
        # First Graph Attention Layer
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads)
        # Second Graph Attention Layer
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(hidden_dim * num_heads, hidden_dim, batch_first=True)
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        # Apply first GAT layer with LeakyReLU activation
        x = self.conv1(x, edge_index)
        x = nn.functional.leaky_relu(x)
        # Apply second GAT layer with LeakyReLU activation
        x = self.conv2(x, edge_index)
        x = nn.functional.leaky_relu(x)
        # Add batch dimension for LSTM
        x = x.unsqueeze(0)
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x)
        # Output layer
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

#===============================================
# Create edges dynamically based on correlation
#===============================================

def create_dynamic_edge_index(features, percentile=PARAMS['correlation_percentile']):
    # Calculate correlation matrix over the last 252 days
    returns = features[[ticker for ticker in valid_tickers]].iloc[-252:]
    correlation_matrix = returns.corr()
    corr_values = correlation_matrix.abs().values.flatten()
    # Determine correlation threshold based on percentile
    corr_threshold = np.percentile(corr_values, percentile)
    edge_index = []
    num_nodes = correlation_matrix.shape[0]
    # Create edges based on correlation threshold
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= corr_threshold:
                    edge_index.append([i, j])
    if not edge_index:
        # If no edges are found, lower the threshold
        corr_threshold = np.percentile(corr_values, percentile - 10)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) >= corr_threshold:
                        edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

#===============
# Training loop
#===============

def train(model, train_data, epochs, learning_rate, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for idx, data in enumerate(train_data):
            data = data.to(device)
            optimizer.zero_grad()
            # Forward pass
            out = model(data.x, data.edge_index)
            # Compute loss
            loss = criterion(out, data.y.mean())
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model

#==========================
# Walk-forward backtesting
#==========================

def walk_forward_backtest(data_list, device):
    predictions = []
    actuals = []
    dates_list = []
    retrain_interval = PARAMS['retrain_interval']
    initial_train_size = PARAMS['initial_train_size']

    print("\nStarting walk-forward backtesting for Graph-Attention Strategy...")
    backtest_progress = tqdm(range(initial_train_size, len(data_list) - retrain_interval, retrain_interval))

    for i in backtest_progress:
        # Define training and test sets
        train_data = data_list[i - initial_train_size:i]
        test_data = data_list[i:i + retrain_interval]

        if len(test_data) == 0:
            break

        # Prepare edge indices using training data only
        train_features = features.iloc[i - initial_train_size:i]
        edge_index = create_dynamic_edge_index(train_features, percentile=PARAMS['correlation_percentile'])

        # Update data objects with new edge_index
        for data_obj in train_data + test_data:
            data_obj.edge_index = edge_index

        # Initialize and train the model
        model = AdvancedGNNModel(num_features=7, hidden_dim=PARAMS['hidden_dim'], num_heads=PARAMS['num_heads'])
        model = train(model, train_data, PARAMS['epochs'], PARAMS['learning_rate'], device)

        # Make predictions on test data
        model.eval()
        with torch.no_grad():
            for data in test_data:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                # Append predictions and actual returns
                predictions.append(out.cpu().numpy())
                actuals.append(data.y.mean().cpu().numpy())
                dates_list.append(data.date)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    dates_list = pd.to_datetime(dates_list)
    return predictions, actuals, dates_list

#==============================
# Metrics to evaluate strategy
#==============================

def evaluate_strategy(predictions, actuals, dates, benchmark_returns, strategy_name):
    # Generate signals based on predictions
    signals = np.sign(predictions)
    # Calculate strategy returns
    strategy_returns = signals * actuals
    strategy_returns = pd.DataFrame({'Returns': strategy_returns}, index=dates)
    # Group by date to get average return per day
    strategy_returns = strategy_returns.groupby(strategy_returns.index).mean()
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    # Calculate Sharpe Ratio
    sharpe_ratio = (strategy_returns['Returns'].mean() / strategy_returns['Returns'].std()) * np.sqrt(252)

    # Align benchmark returns
    benchmark_returns = benchmark_returns[strategy_returns.index[0]:strategy_returns.index[-1]]
    benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1

    # Calculate Alpha and Beta
    aligned_benchmark_returns = benchmark_returns.loc[strategy_returns.index]
    cov_matrix = np.cov(strategy_returns['Returns'], aligned_benchmark_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha = (strategy_returns['Returns'].mean() - beta * aligned_benchmark_returns.mean()) * 252

    metrics = {
        'Strategy': strategy_name,
        'Alpha': alpha,
        'Beta': beta,
        'Sharpe Ratio': sharpe_ratio
    }

    return cumulative_returns, benchmark_cum_returns, metrics, strategy_returns

#=============================
# Run backtest and evaluation
#=============================

# Run the backtest
predictions, actuals, dates = walk_forward_backtest(data_list, PARAMS['device'])
# Evaluate the strategy
cumulative_returns, benchmark_cum_returns, metrics, strategy_returns = evaluate_strategy(predictions, actuals, dates, benchmark, 'Graph-Attention Strategy')

#=========================
# Plot cumulative returns
#========================

tech_index_aligned = tech_index[cumulative_returns.index[0]:cumulative_returns.index[-1]]
tech_cum_returns = (1 + tech_index_aligned).cumprod() - 1

plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")
colors = ["#1f77b4", "#2ca02c", "#9467bd"]  # Blue, Green, Purple
plt.plot(cumulative_returns.index, cumulative_returns['Returns'], label='Graph-Attention Strategy', linewidth=2, color=colors[0])
plt.plot(benchmark_cum_returns.index, benchmark_cum_returns['Benchmark'], label='Benchmark (S&P 500)', linewidth=2, color=colors[1])
plt.plot(tech_cum_returns.index, tech_cum_returns['Tech Fund'], label='Technology Index Fund (QQQ)', linewidth=2, color=colors[2])
plt.title('Cumulative Returns Comparison', fontsize=16, fontweight='bold', color='#333333')
plt.xlabel('Date', fontsize=14, color='#555555')
plt.ylabel('Cumulative Return', fontsize=14, color='#555555')
plt.grid(visible=True, linestyle='--', linewidth=0.5, color='gray')
plt.legend(fontsize=12, frameon=False)
plt.xticks(fontsize=12, color='#555555')
plt.yticks(fontsize=12, color='#555555')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('returns.png', dpi=600)

#=================
# Display metrics
#=================

metrics_data = [[metrics['Strategy'], metrics['Alpha'], metrics['Beta'], metrics['Sharpe Ratio']]]
headers = ['Strategy', 'Alpha', 'Beta', 'Sharpe Ratio']
print(tabulate(metrics_data, headers=headers, tablefmt='grid'))
plt.show()