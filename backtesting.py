import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def backtest_strategy(df, signal_column, title):
    returns = pd.DataFrame()
    returns['Price'] = df['Actual_Price']
    returns['Signal'] = df[signal_column]
    returns['Return'] = returns['Price'].pct_change()
    returns['Strategy_Return'] = returns['Signal'].shift(1) * returns['Return']
    returns['Cumulative_Return'] = (1 + returns['Strategy_Return'].fillna(0)).cumprod()
    return returns

def calculate_performance_metrics(strategy_returns, risk_free_rate=0.01):
    daily_returns = strategy_returns['Strategy_Return'].dropna()
    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan
    return annualized_return, annualized_volatility, sharpe_ratio

def plot_cumulative_returns(returns_list, labels, title):
    plt.figure(figsize=(14, 7))
    for returns, label in zip(returns_list, labels):
        plt.plot(returns.index, returns['Cumulative_Return'], label=label)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()