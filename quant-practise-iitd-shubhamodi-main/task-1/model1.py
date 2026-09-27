# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:56:06 2024
@author: Shubham Modi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ta  # For technical analysis indicators

# Feature Engineering
def engineer_features(df):
    """
    Add technical indicators and other relevant features for trading signals.
    This helps in constructing features like Simple Moving Average (SMA), 
    Relative Strength Index (RSI), etc., which are essential in predicting price movements.
    """
    df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)  # 10-period SMA
    df['SMA_30'] = ta.trend.sma_indicator(df['close'], window=30)  # 30-period SMA
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)  # Relative Strength Index (RSI)
    df['MACD'] = ta.trend.macd_diff(df['close'])  # Moving Average Convergence Divergence (MACD)
    
    # Price spread between high and low, and open and close
    df['HL_spread'] = df['high'] - df['low']  
    df['OC_spread'] = df['close'] - df['open']  
    
    # Momentum: today's close - yesterday's close
    df['momentum'] = df['close'] - df['close'].shift(1)  
    
    # Percentage change in price from the previous day (returns)
    df['returns'] = df['close'].pct_change()
    
    # Target variable for supervised learning: 1 if next day's close is higher, 0 otherwise
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    # Drop rows with NaN values (due to shifting or rolling calculations)
    return df.dropna()

# Risk Management and Position Sizing
def calculate_position_size(df, risk_per_trade=0.01):
    """
    Calculate position size based on risk management. The risk per trade is a fixed percentage
    of the portfolio value. The stop loss level is used to determine how much risk is involved in the trade.
    """
    # Risk per trade based on stop loss (difference between stop loss price and entry price)
    df['risk'] = df['stop_loss_price'] - df['close']
    
    # Position size is calculated by dividing the risk capital by the risk per trade.
    df['position_size'] = (df['risk'] * df['close']) / (risk_per_trade * df['portfolio_value'])
    
    # Ensure a minimum position size of 1 to avoid overly small trades
    df['position_size'] = df['position_size'].clip(lower=1)
    return df

# Entry and Exit Signals
def generate_signals(df):
    """
    Generate buy and sell signals based on technical indicators like SMA and RSI.
    Buy signal: If the short-term SMA crosses above the long-term SMA and RSI is oversold (<30).
    Sell signal: If the short-term SMA crosses below the long-term SMA and RSI is overbought (>70).
    """
    df['signal'] = 0  # Initialize signal column with 0 (no position)
    
    # Buy when the 10-period SMA is above the 30-period SMA and RSI is below 30 (oversold condition)
    df.loc[(df['SMA_10'] > df['SMA_30']) & (df['RSI'] < 30), 'signal'] = 1  # Buy signal
    
    # Sell when the 10-period SMA is below the 30-period SMA and RSI is above 70 (overbought condition)
    df.loc[(df['SMA_10'] < df['SMA_30']) & (df['RSI'] > 70), 'signal'] = -1  # Sell signal
    return df

# Sharpe Ratio
def calculate_sharpe_ratio(df):
    """
    Sharpe ratio measures risk-adjusted return. It's the ratio of the mean of strategy returns 
    over their standard deviation, annualized by multiplying by the square root of 252 (number of trading days).
    """
    sharpe_ratio = np.sqrt(252) * df['trade_returns'].mean() / df['trade_returns'].std()
    return sharpe_ratio

# Sortino Ratio
def calculate_sortino_ratio(df):
    """
    Sortino ratio is similar to Sharpe ratio but considers only downside volatility (negative returns).
    It gives a more focused view on the risk of negative outcomes.
    """
    target_return = 0  # Set target return to 0 for simplicity
    downside_returns = df.loc[df['trade_returns'] < target_return, 'trade_returns']  # Only consider negative returns
    sortino_ratio = np.sqrt(252) * df['trade_returns'].mean() / downside_returns.std()
    return sortino_ratio

# Max Drawdown
def calculate_max_drawdown(df):
    """
    Max Drawdown calculates the maximum observed loss from a peak to a trough in portfolio value.
    It measures the worst case risk of the strategy.
    """
    max_drawdown = (df['portfolio_value'] / df['portfolio_value'].cummax() - 1).min()
    return max_drawdown

# Annualized Return
def calculate_annualized_return(df):
    """
    Annualized return calculates the equivalent yearly return from the cumulative return.
    It's the compound return per year based on daily data.
    """
    total_return = df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1
    
    # Ensure no division by zero and calculate based on the number of trading days
    if len(df) > 1 and df['portfolio_value'].iloc[0] > 0:
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
    else:
        annualized_return = 0
    return annualized_return

# Cumulative Return
def calculate_cumulative_return(df):
    """
    Cumulative return calculates the total return from the start of the period to the end.
    """
    if df['portfolio_value'].iloc[0] > 0:  # Prevent division by zero
        cumulative_return = df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1
    else:
        cumulative_return = 0
    return cumulative_return

# Backtesting the Strategy
def backtest_strategy(df, initial_capital=10000):
    """
    This function runs the backtest by simulating portfolio value over time based on 
    the generated signals. It calculates key performance metrics such as Sharpe ratio,
    Sortino ratio, and Max Drawdown.
    """
    df['portfolio_value'] = initial_capital  # Start with an initial capital of $10,000
    df['position'] = df['signal'].shift(1)  # Shift signals to reflect prior day's decision
    df['trade_returns'] = df['position'] * df['returns']  # Calculate strategy returns based on positions taken
    
    # Cumulative return of the strategy, updating portfolio value with each trade
    df['portfolio_value'] *= (1 + df['trade_returns']).cumprod()
    
    # Drop rows with NaN values to avoid errors during calculations
    df = df.dropna(subset=['portfolio_value', 'trade_returns'])
    
    # Calculate performance metrics
    sharpe_ratio = calculate_sharpe_ratio(df)
    sortino_ratio = calculate_sortino_ratio(df)
    max_drawdown = calculate_max_drawdown(df)
    annualized_return = calculate_annualized_return(df)
    cumulative_return = calculate_cumulative_return(df)
    
    # Return a dictionary of key performance metrics
    return {
        'final_value': df['portfolio_value'].iloc[-1],
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'annualized_return': annualized_return,
        'cumulative_return': cumulative_return
    }

# Main Execution
if __name__ == "__main__":
    # Load Data
    df = pd.read_csv('D:\\2cents\\datasets\\quant\\ml\\ETHUSDT_1m.csv')  # Adjust nrows for large datasets
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp to datetime format
    df.set_index('timestamp', inplace=True)  # Set timestamp as the index for easy time-based analysis
    
    # Feature Engineering
    df = engineer_features(df)
    
    # Generate Signals
    df = generate_signals(df)
    
    # Risk Management
    df['stop_loss_price'] = df['close'] * 0.98  # Assume a 2% stop loss from the entry price
    df['portfolio_value'] = 0  # Initialize portfolio_value to avoid KeyError
    df = calculate_position_size(df)
    
    # Backtesting
    results = backtest_strategy(df)
    
    # Display Results
    print("Final Portfolio Value: ${:.2f}".format(results['final_value']))
    print("Sharpe Ratio: {:.4f}".format(results['sharpe_ratio']))
    print("Sortino Ratio: {:.4f}".format(results['sortino_ratio']))
    print("Max Drawdown: {:.4f}".format(results['max_drawdown']))
    print("Annualized Return: {:.2%}".format(results['annualized_return']))
    print("Cumulative Return: {:.2%}".format(results['cumulative_return']))
    
    # Plot Portfolio Value over time
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['portfolio_value'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.show()
