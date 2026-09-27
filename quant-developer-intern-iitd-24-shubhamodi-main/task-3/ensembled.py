import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from scipy.stats import gmean
from sklearn.preprocessing import StandardScaler

# Calculate daily returns: (Price_t - Price_(t-1)) / Price_(t-1)
@jit(nopython=True)  # JIT compilation for faster execution
def calculate_returns(prices):
    return np.diff(prices) / prices[:-1]

# Rolling mean (moving average) calculation over a window
@jit(nopython=True)
def calculate_rolling_mean(arr, window):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        if i < window:
            result[i] = np.mean(arr[:i+1])  # Partial mean for the first 'window' elements
        else:
            result[i] = np.mean(arr[i-window+1:i+1])  # Full window mean for the rest
    return result

# Rolling standard deviation calculation over a window
@jit(nopython=True)
def calculate_rolling_std(arr, window):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        if i < window:
            result[i] = np.std(arr[:i+1]) if i > 0 else 0  # Partial std deviation for early elements
        else:
            result[i] = np.std(arr[i-window+1:i+1])  # Full window std for the rest
    return result

# RSI (Relative Strength Index) calculation
def rsi(prices, period=14):
    returns = np.diff(prices)  # Price changes
    up = np.maximum(returns, 0)  # Gains (positive returns)
    down = -np.minimum(returns, 0)  # Losses (negative returns)
    
    avg_gain = calculate_rolling_mean(up, period)  # Rolling mean of gains
    avg_loss = calculate_rolling_mean(down, period)  # Rolling mean of losses
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)  # Calculate RS (Relative Strength)
    rsi = 100 - (100 / (1 + rs))  # Final RSI formula
    return np.append(np.zeros(1), rsi)  # Add a zero at the beginning to align with price data

# MACD (Moving Average Convergence Divergence) calculation
def macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calculate_rolling_mean(prices, fast)  # Short-term EMA (fast)
    ema_slow = calculate_rolling_mean(prices, slow)  # Long-term EMA (slow)
    
    macd_line = ema_fast - ema_slow  # MACD line (difference of EMAs)
    signal_line = calculate_rolling_mean(macd_line, signal)  # Signal line (EMA of MACD line)
    
    return macd_line - signal_line  # MACD histogram (MACD line - signal line)

# Bollinger Bands calculation
def bollinger_bands(prices, window=20, num_std=2):
    sma = calculate_rolling_mean(prices, window)  # Simple moving average (SMA)
    std = calculate_rolling_std(prices, window)  # Rolling standard deviation
    
    upper = sma + num_std * std  # Upper Bollinger Band (SMA + N * std)
    lower = sma - num_std * std  # Lower Bollinger Band (SMA - N * std)
    
    # Bollinger Band index (price relative to the band)
    bb = (prices - sma) / (upper - lower)
    
    return np.where(np.isfinite(bb), bb, 0)  # Replace inf/nan values with 0

# Ensemble alpha strategy: combine RSI, MACD, and Bollinger Band signals
def ensemble_alpha(prices, lookback=20):
    rsi_signal = rsi(prices, lookback) - 50  # Center RSI around 0 (50 is the neutral value in RSI)
    macd_signal = macd(prices)  # MACD signal
    bb_signal = bollinger_bands(prices, lookback)  # Bollinger Bands signal
    
    # Stack signals into a matrix
    alphas = np.column_stack((rsi_signal, macd_signal, bb_signal))
    
    # Normalize the signals using StandardScaler
    scaler = StandardScaler()
    normalized_alphas = scaler.fit_transform(alphas)
    
    # Calculate dynamic weights based on recent performance (mean of the last 'lookback' periods)
    weights = np.abs(normalized_alphas[-lookback:].mean(axis=0))
    weights = np.where(np.isfinite(weights), weights, 0)  # Replace any inf/nan with 0
    weights_sum = np.sum(weights)
    
    # Normalize the weights if their sum is non-zero
    if weights_sum > 0:
        weights /= weights_sum
    else:
        weights = np.ones_like(weights) / len(weights)  # Equal weights if invalid
    
    # Final alpha signal is a weighted sum of normalized signals
    return np.dot(normalized_alphas, weights)

# Backtest strategy using the generated alpha signal
@jit(nopython=True)
def backtest(prices, alpha_signal, transaction_cost=0.0005):
    signals = np.sign(alpha_signal)  # Trade signals based on alpha (long/short)
    returns = calculate_returns(prices)  # Asset returns
    
    # Strategy returns (buy/sell signal * returns) - transaction cost
    strategy_returns = signals[:-1] * returns - np.abs(np.diff(signals)) * transaction_cost
    return strategy_returns

# Generate performance report (total return, annualized return, Sharpe ratio, and drawdowns)
def generate_performance_report(strategy_returns):
    total_return = np.prod(1 + strategy_returns) - 1  # Total return
    annualized_return = gmean(1 + strategy_returns) ** 252 - 1  # Annualized return (assuming 252 trading days)
    sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)  # Sharpe ratio
    cumulative_returns = np.cumprod(1 + strategy_returns)  # Cumulative returns
    
    # Calculate drawdowns (peak-to-trough declines)
    drawdowns = cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1
    max_drawdown = np.min(drawdowns)  # Maximum drawdown
    
    # Print the performance metrics
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns)
    plt.title("Cumulative Strategy Returns")
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Returns")
    plt.show()

# Load and preprocess data (read from CSV and resample daily prices)
def load_and_preprocess_data(file_path, date_col='timestamp', price_col='close'):
    data = pd.read_csv(file_path, usecols=[date_col, price_col], parse_dates=[date_col], index_col=date_col)
    daily_data = data[price_col].resample('D').last().ffill()  # Resample to daily frequency
    return daily_data.values

# Plot the alpha signal and asset price
def plot_alpha_signal(prices, alpha_signal):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot asset price
    ax1.plot(prices)
    ax1.set_title('Asset Price')
    ax1.set_ylabel('Price')
    
    # Plot alpha signal
    ax2.plot(alpha_signal)
    ax2.set_title('Alpha Signal')
    ax2.set_ylabel('Signal Strength')
    ax2.set_xlabel('Trading Days')
    
    plt.tight_layout()
    plt.show()

# Optimize lookback period to maximize the Sharpe ratio
def optimize_lookback(prices, lookback_range):
    best_sharpe = -np.inf  # Initialize best Sharpe ratio
    best_lookback = 0  # Initialize best lookback period
    
    for lookback in lookback_range:
        alpha_signal = ensemble_alpha(prices, lookback)  # Generate alpha signal for the lookback
        strategy_returns = backtest(prices, alpha_signal)  # Backtest the strategy
        
        # Calculate Sharpe ratio for the strategy
        sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        
        # Track the best Sharpe ratio and lookback period
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_lookback = lookback
    
    return best_lookback

# Main function to load data, optimize lookback, and backtest the strategy
def main():
    file_path = 'D:\\2cents\\datasets\\quant\\ml\\BTCUSDT_1m.csv'  # Replace with the path to your dataset
    
    prices = load_and_preprocess_data(file_path)  # Load and preprocess data
    
    # Optimize lookback period
    lookback_range = range(5, 100, 5)
    best_lookback = optimize_lookback(prices, lookback_range)
    print(f"Best lookback period: {best_lookback}")
    
    # Generate alpha signal using the best lookback period
    alpha_signal = ensemble_alpha(prices, best_lookback)
    
    # Ensure the alpha signal length matches the price length
    if len(alpha_signal) != len(prices):
        raise ValueError("Alpha signal length does not match prices length")
    
    plot_alpha_signal(prices, alpha_signal)  # Plot the alpha signal
    
    strategy_returns = backtest(prices, alpha_signal)  # Backtest the strategy
    generate_performance_report(strategy_returns)  # Generate performance report

# Run the main function
if __name__ == "__main__":
    main()
