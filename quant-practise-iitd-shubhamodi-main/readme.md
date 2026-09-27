Trading Strategy and Market Analysis
This repository contains three major tasks related to technical analysis and trading strategies using Python. The tasks include working with stock or cryptocurrency market data, calculating key indicators like RSI, MACD, Bollinger Bands, and backtesting strategies to generate performance metrics.

Requirements
Make sure you have the following packages installed before running the code:

# Trading Strategy Analysis and Backtesting

This repository contains Python-based implementations for trading strategies and technical analysis indicators, along with a backtesting framework.

## Prerequisites

To run the code in this repository, you need to install the following Python libraries:

```bash
pip install numpy pandas matplotlib numba scipy scikit-learn

```bash
##pip install numpy pandas matplotlib numba scipy scikit-learn
Task 1: Loading Historical Data and Basic Performance Metrics
This task involves loading historical market data and calculating basic performance metrics.

Steps:

Load CSV File: Reads market data (like BTC/USDT) from a CSV file.
Data Processing: Preprocesses the data, including calculating daily returns and resampling to a daily frequency.
Performance Metrics: Calculates and prints performance metrics such as total return, annualized return, Sharpe ratio, and maximum drawdown.
Execution:
Place your CSV file in the specified folder or path.
Run the script using:
bash
Copy code
python task_1.py
Task 2: Technical Indicators and Trading Signals
This task involves calculating popular technical analysis indicators such as:

RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
Bollinger Bands
Steps:

RSI: Calculated using a rolling average of gains and losses over a given period (default: 14).
MACD: Uses the difference between fast and slow exponential moving averages (EMA).
Bollinger Bands: Uses a simple moving average (SMA) and standard deviation for upper and lower bands.
Execution:
To calculate these indicators and visualize them, run:

bash
Copy code
python task_2.py
You will get visual plots for each indicator, along with the calculated signal strengths.

Task 3: Ensemble Alpha Strategy and Backtesting
In this task, we implement a more advanced ensemble alpha trading strategy by combining multiple technical indicators and backtesting them.

Key Functions:

Ensemble Alpha: Combines RSI, MACD, and Bollinger Bands signals, normalizes them, and assigns dynamic weights based on recent performance.
Backtesting: Tests the strategy over historical data and calculates returns based on trading signals.
Performance Report: Generates a performance report that includes total return, annualized return, Sharpe ratio, and maximum drawdown. It also plots cumulative strategy returns.
Execution:
Specify the path to your historical data CSV file.
Run the following command to execute the strategy:

bash code
python task_3.py
This will generate alpha signals, backtest the strategy, and produce a performance report along with plots.

Data Format
The data used for these tasks should be in CSV format and include at least two columns:

timestamp: Date and time of the price data.
close: Closing price for each period.
Make sure your dataset follows this structure for smooth execution.

Code Optimization
To enhance performance, Numba is used to speed up the calculation of returns, rolling averages, and backtesting logic.

Installing Numba
bash code
pip install numba
License
This project is licensed under the MIT License. Feel free to use and modify the code
