# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:56:06 2024

@author: shubh
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import ta  # Technical analysis library for feature generation
import matplotlib.pyplot as plt

# 1. Feature Engineering
def engineer_features(df):
    """
    This function adds several technical indicators and statistical features
    to the DataFrame, which will later be used as features for the machine learning model.
    """
    # Technical Indicators: Simple Moving Averages (SMA), Relative Strength Index (RSI), MACD, and Bollinger Bands
    df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['SMA_30'] = ta.trend.sma_indicator(df['close'], window=30)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = (
        ta.volatility.bollinger_hband_indicator(df['close']), 
        ta.volatility.bollinger_mavg(df['close']), 
        ta.volatility.bollinger_lband_indicator(df['close'])
    )
    
    # Price Action Features: High-Low Spread, Open-Close Spread, Momentum, and Volatility
    df['HL_spread'] = df['high'] - df['low']
    df['OC_spread'] = df['close'] - df['open']
    df['momentum'] = df['close'] - df['close'].shift(1)  # Difference in closing prices
    df['volatility'] = df['close'].rolling(window=10).std()  # Rolling standard deviation (volatility)

    # Statistical Features: Rolling mean, standard deviation, skewness, and kurtosis
    df['rolling_mean'] = df['close'].rolling(window=20).mean()
    df['rolling_std'] = df['close'].rolling(window=20).std()
    df['rolling_skew'] = df['close'].rolling(window=20).skew()
    df['rolling_kurt'] = df['close'].rolling(window=20).kurt()
    
    # Lagged Features: Closing price and volume for the past 5 periods (lags)
    for i in range(1, 6):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
    
    # Target variable: 1 if next day's close is higher, 0 otherwise
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    # Return the DataFrame after removing NaN values (resulting from shifting and rolling operations)
    return df.dropna()

# 2. Normalization and Standardization
def normalize_features(df, feature_columns):
    """
    Normalize the selected feature columns using StandardScaler, which standardizes
    features by removing the mean and scaling to unit variance.
    """
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

# 3. Machine Learning Model Development and Backtesting
def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model on the training data to predict the target variable.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)  # Train the model with the training data
    return model

def backtest_strategy(df, model, feature_columns, initial_capital=10000):
    """
    Simulate backtesting by using the trained model to make predictions on the given data.
    Calculate key performance metrics like Sharpe Ratio, Max Drawdown, Total Return, Accuracy, etc.
    """
    df['prediction'] = model.predict(df[feature_columns])  # Use the model to predict the target
    df['position'] = df['prediction'].shift(1)  # Lag the predictions to simulate taking action after prediction
    df['returns'] = df['close'].pct_change()  # Daily percentage change in closing prices
    df['strategy_returns'] = df['position'] * df['returns']  # Strategy returns based on positions taken
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()  # Cumulative returns of the strategy
    df['cumulative_strategy_returns'] = initial_capital * df['cumulative_returns']  # Portfolio value based on strategy

    # Calculate key metrics: Sharpe Ratio, Max Drawdown, and Total Return
    sharpe_ratio = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
    max_drawdown = (df['cumulative_strategy_returns'] / df['cumulative_strategy_returns'].cummax() - 1).min()
    total_return = df['cumulative_returns'].iloc[-1] - 1

    # Calculate performance metrics and return results as a dictionary
    return {
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Total Return': total_return,
        'Accuracy': accuracy_score(df['target'], df['prediction']),
        'Precision': precision_score(df['target'], df['prediction']),
        'Recall': recall_score(df['target'], df['prediction'])
    }

# 4. Validation and Out-of-Sample Testing
def validate_model(df, model, scaler, feature_columns):
    """
    Validate the model by splitting the data into training, validation, and test sets,
    backtesting each set and comparing their performance.
    """
    # Split the data into training (60%), validation (20%), and test (20%) sets
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)

    # Normalize the features for all datasets using the previously fitted scaler
    train_data[feature_columns] = scaler.transform(train_data[feature_columns])
    val_data[feature_columns] = scaler.transform(val_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    # Backtest on training, validation, and test datasets and return the results
    train_results = backtest_strategy(train_data, model, feature_columns)
    val_results = backtest_strategy(val_data, model, feature_columns)
    test_results = backtest_strategy(test_data, model, feature_columns)

    return train_results, val_results, test_results

# 5. Strategy Development
def develop_strategy(df, model, feature_columns):
    """
    Develop the final trading strategy by generating entry and exit signals based on
    the model predictions, while incorporating risk management like stop-loss and position sizing.
    """
    # Predict signals using the model
    df['signal'] = model.predict(df[feature_columns])
    df['position'] = df['signal'].shift(1)  # Shift the signal to reflect decisions made after predictions
    df['returns'] = df['close'].pct_change()  # Daily returns
    df['strategy_returns'] = df['position'] * df['returns']  # Strategy returns based on the position
    
    # Entry and exit signals for the strategy
    df['entry'] = (df['signal'] == 1) & (df['signal'].shift(1) == 0)  # Buy signal when the model predicts 1
    df['exit'] = (df['signal'] == 0) & (df['signal'].shift(1) == 1)   # Sell signal when the model predicts 0
    
    # Risk management: Implement a 2% stop-loss
    stop_loss = 0.02
    df['stop_loss_price'] = np.where(df['position'] == 1, df['close'] * (1 - stop_loss), np.nan)
    df['stop_loss_triggered'] = (df['position'] == 1) & (df['low'] <= df['stop_loss_price'])  # Check if stop-loss is hit
    
    # Position sizing based on volatility (lower volatility -> larger position size)
    df['volatility'] = df['returns'].rolling(window=20).std()  # Rolling 20-period volatility
    df['position_size'] = 1 / df['volatility']  # Inverse volatility to determine position size
    df['position_size'] = df['position_size'] / df['position_size'].rolling(window=20).mean()  # Normalize
    df['position_size'] = df['position_size'].clip(0.5, 2)  # Limit position size between 0.5x and 2x
    
    return df

# Main execution
if __name__ == "__main__":
    # Load data (replace with your actual data file path)
    df = pd.read_csv('D:\\2cents\\datasets\\quant\\ml\\SOLUSDT_1m.csv', nrows=100000)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp to datetime
    df = df.set_index('timestamp')  # Set the timestamp as the index

    # 1. Feature Engineering
    df = engineer_features(df)
    
    # Define feature columns to avoid inconsistency (exclude non-feature columns)
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'target', 'prediction', 'position', 'cumulative_returns', 'cumulative_strategy_returns']]
    
    # 2. Normalization and Standardization
    df, scaler = normalize_features(df, feature_columns)
    
    # Split data into features (X) and target (y)
    X = df[feature_columns]  # Use only the feature columns
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 3. Machine Learning Model Development and Backtesting
    model = train_model(X_train, y_train)
    backtest_results = backtest_strategy(df, model, feature_columns)
    
    # Display the backtest results
    print("Backtest Results:")
    for metric, value in backtest_results.items():
        print(f"{metric}: {value:.4f}")
    
    # 4. Validation and Out-of-Sample Testing
    train_results, val_results, test_results = validate_model(df, model, scaler, feature_columns)
    
    # Display the validation results
    print("\nValidation Results:")
    print("Training Set:")
    for metric, value in train_results.items():
        print(f"{metric}: {value:.4f}")
    print("\nValidation Set:")
    for metric, value in val_results.items():
        print(f"{metric}: {value:.4f}")
    print("\nTest Set:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    # 5. Strategy Development
    final_strategy = develop_strategy(df, model, feature_columns)
    
    # Plot strategy performance: compare strategy returns with buy-and-hold returns
    plt.figure(figsize=(12, 6))
    plt.plot(final_strategy.index, final_strategy['cumulative_returns'], label='Strategy Returns')
    plt.plot(final_strategy.index, (1 + final_strategy['returns']).cumprod(), label='Buy and Hold Returns')
    plt.title('Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()
