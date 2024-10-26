# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from scipy.stats import linregress
from models import RNNModel, LSTMModel, MomentumMeanReversionModel
from backtesting import backtest_strategy, calculate_performance_metrics, plot_cumulative_returns
from utils import StockDataset, load_data, create_sequences, get_scaler
import matplotlib.pyplot as plt

# Parameters
tickers = ['GS', 'JPM']
start_date = '2001-01-01'
end_date = '2023-10-25'
lookback = 60
retrain_window = 60
epochs = 10
batch_size = 16
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
risk_free_rate = 0.01

# Load data
data = load_data(tickers, start_date, end_date)

# Prepare for benchmarking with CAPM
market_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
market_returns = market_data.pct_change().dropna()

# Initialize results list
results = []

# Ensure timezone-naive indices for both market and stock returns
market_returns.index = market_returns.index.tz_localize(None)

# Loop through each ticker
for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    scaler, scaled_data = get_scaler(data[ticker])

    predictions_rnn = []
    predictions_lstm = []
    actual_prices = []
    dates = []
    mmr_signals = []

    # Calculate stock returns
    stock_returns = data[ticker].pct_change().dropna()
    stock_returns.index = stock_returns.index.tz_localize(None)

    # Align market and stock returns using an 'inner' join to avoid data mismatches
    aligned_data = pd.DataFrame({
        'market_returns': market_returns,
        'stock_returns': stock_returns
    }).dropna()

    # Remove non-finite values in aligned data
    aligned_data = aligned_data[np.isfinite(aligned_data['market_returns']) & np.isfinite(aligned_data['stock_returns'])]

    # Check for variance in the aligned market data and calculate CAPM
    try:
        if len(aligned_data) > 1 and np.var(aligned_data['market_returns']) > 0:
            slope, intercept, _, _, _ = linregress(aligned_data['market_returns'], aligned_data['stock_returns'])
            beta = slope
            alpha = intercept
            expected_return_capm = risk_free_rate + beta * (market_returns.mean() * 252 - risk_free_rate)
            print(f"CAPM Expected Annual Return for {ticker}: {expected_return_capm:.2%}, Beta: {beta:.2f}")
        else:
            print(f"Constant or insufficient data for CAPM calculation for {ticker}")
            beta, alpha, expected_return_capm = np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Error in CAPM calculation for {ticker}: {e}")
        beta, alpha, expected_return_capm = np.nan, np.nan, np.nan

    # Buy and Hold Strategy
    buy_hold_returns = (1 + stock_returns).cumprod()

    # Initialize Momentum Mean Reversion Model
    mmr_model = MomentumMeanReversionModel(momentum_lookback=5, ma_window=20)
    mmr_signal_full = mmr_model.generate_signals(data[ticker])

    # Walk-forward validation with nested progress bars for training steps
    rnn_model = RNNModel().to(device)
    lstm_model = LSTMModel().to(device)
    total_steps = len(scaled_data) - lookback - 1

    # Outer progress bar for each walk-forward validation step
    with tqdm(total=total_steps, desc=f'{ticker} Walk-forward Validation', position=0) as outer_pbar:
        for i in range(lookback, len(scaled_data) - 1):
            # Update progress description in outer loop
            outer_pbar.set_postfix({'Step': f"{i - lookback + 1}/{total_steps}"})

            if (i - lookback) % retrain_window == 0:
                # Prepare training data
                train_sequences = create_sequences(scaled_data[:i], lookback)
                train_dataset = StockDataset(train_sequences)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

                if len(train_loader) == 0:
                    # Skipping training due to insufficient data
                    continue

                # Middle progress bar for epochs
                with tqdm(total=epochs, desc="RNN Training Epochs", position=1, leave=False) as epoch_pbar:
                    for epoch in range(epochs):
                        epoch_loss = 0
                        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", position=2, leave=False) as batch_pbar:
                            for seq, labels in train_loader:
                                seq, labels = seq.to(device), labels.to(device)
                                rnn_model.optimizer.zero_grad()
                                y_pred = rnn_model(seq)
                                loss = rnn_model.criterion(y_pred.view(-1), labels.view(-1))
                                loss.backward()
                                rnn_model.optimizer.step()
                                epoch_loss += loss.item()
                                batch_pbar.set_postfix(loss=epoch_loss / (batch_pbar.n + 1))
                                batch_pbar.update(1)
                        epoch_pbar.update(1)

                # Train LSTM Model similarly
                with tqdm(total=epochs, desc="LSTM Training Epochs", position=1, leave=False) as epoch_pbar:
                    for epoch in range(epochs):
                        epoch_loss = 0
                        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", position=2, leave=False) as batch_pbar:
                            for seq, labels in train_loader:
                                seq, labels = seq.to(device), labels.to(device)
                                lstm_model.optimizer.zero_grad()
                                y_pred = lstm_model(seq)
                                loss = lstm_model.criterion(y_pred.view(-1), labels.view(-1))
                                loss.backward()
                                lstm_model.optimizer.step()
                                epoch_loss += loss.item()
                                batch_pbar.set_postfix(loss=epoch_loss / (batch_pbar.n + 1))
                                batch_pbar.update(1)
                        epoch_pbar.update(1)

            # Prepare test input and make predictions
            test_seq = scaled_data[i - lookback:i]
            test_seq = torch.tensor(test_seq, dtype=torch.float32).unsqueeze(0)

            # Get actual price
            actual_price = data[ticker].values[i]
            actual_prices.append(actual_price)
            dates.append(data.index[i])

            # Predictions with RNN and LSTM
            pred_rnn = rnn_model.predict(test_seq, device)
            pred_rnn = scaler.inverse_transform(pred_rnn)
            predictions_rnn.append(pred_rnn.flatten()[0])

            pred_lstm = lstm_model.predict(test_seq, device)
            pred_lstm = scaler.inverse_transform(pred_lstm)
            predictions_lstm.append(pred_lstm.flatten()[0])

            # Get MMR signal
            mmr_signal = mmr_signal_full.iloc[i]
            mmr_signals.append(mmr_signal)

            # Update outer progress bar
            outer_pbar.update(1)

    # Create DataFrame for predictions and actual prices
    df_predictions = pd.DataFrame({
        'Date': dates,
        'Actual_Price': actual_prices,
        'RNN_Prediction': predictions_rnn,
        'LSTM_Prediction': predictions_lstm,
        'MMR_Signal': mmr_signals
    })
    df_predictions.set_index('Date', inplace=True)

    # Generate trading signals
    df_predictions['RNN_Signal'] = np.where(df_predictions['RNN_Prediction'] > df_predictions['Actual_Price'].shift(1), 1, -1)
    df_predictions['LSTM_Signal'] = np.where(df_predictions['LSTM_Prediction'] > df_predictions['Actual_Price'].shift(1), 1, -1)
    df_predictions['Buy_and_Hold_Signal'] = 1  # Always long

    # Backtest strategies
    rnn_returns = backtest_strategy(df_predictions, 'RNN_Signal', f'{ticker} RNN Strategy')
    lstm_returns = backtest_strategy(df_predictions, 'LSTM_Signal', f'{ticker} LSTM Strategy')
    mmr_returns = backtest_strategy(df_predictions, 'MMR_Signal', f'{ticker} MMR Strategy')
    buy_hold = backtest_strategy(df_predictions, 'Buy_and_Hold_Signal', f'{ticker} Buy and Hold Strategy')

    # Plot cumulative returns
    plot_cumulative_returns(
        [rnn_returns, lstm_returns, mmr_returns, buy_hold],
        [f'{ticker} RNN Strategy', f'{ticker} LSTM Strategy', f'{ticker} MMR Strategy', f'{ticker} Buy and Hold'],
        f'Cumulative Returns Comparison for {ticker}'
    )

    # Calculate performance metrics
    for strategy_name, strategy_returns in zip(
        ['RNN', 'LSTM', 'MMR', 'Buy and Hold'],
        [rnn_returns, lstm_returns, mmr_returns, buy_hold]
    ):
        annualized_return, annualized_volatility, sharpe_ratio = calculate_performance_metrics(strategy_returns, risk_free_rate)
        alpha = annualized_return - expected_return_capm
        results.append([
            f'{ticker} {strategy_name}',
            f"{alpha:.4%}",
            f"{beta:.2f}",
            f"{annualized_return:.2%}",
            f"{annualized_volatility:.2%}",
            f"{sharpe_ratio:.2f}"
        ])

# Display performance metrics
print("\nPerformance Metrics:")
print(tabulate(results, headers=["Strategy", "Alpha", "Beta", "Annualized Return", "Annualized Volatility", "Sharpe Ratio"], tablefmt="grid"))