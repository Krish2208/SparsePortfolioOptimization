from markowitz_cvx_rohan import MarkowitzOptimization as Markowitz
import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle

benchmark_index = False

# Define the tickers and time period
tickers = ["SO", "KMB", "K", "AZO", "GIS", "HSY",
           "PEP", "WMT", "NEM", "MCD", "AAPL", "CPB",
           "PG", "CAG", "FOH", "PPL", "DUK", "SYY",
           "REGN", "ABT", "JNJ", "ORLY", "MNST", "SHW",
           "THC", "GILD", "PG", "LLY", "BIIB"]

start_date = "2007-12-31"
end_date = "2012-12-31"

if os.path.exists("data.csv"):
    with open("data.csv", "rb") as f:
        data = pickle.load(f)
else:
    data = yf.download(tickers, start=start_date, end=end_date)
    with open("data.csv", "wb") as f:
        pickle.dump(data, f)
        
if benchmark_index:
    # Get the data for the index (S&P 500)
    index_ticker = "^GSPC"
    index_data = yf.download(index_ticker, start=start_date, end=end_date)
    # Calculate the target return from the index price
    target_return = index_data['Adj Close'].pct_change().mean()
    print("Target Return:")
    print(target_return)


# Calculate the covariance matrix
cov_matrix = data['Adj Close'].pct_change()
print("Covariance Matrix:")
print(cov_matrix)
# lambda x: pd.Series(x).fillna(x.mean())).cov()
print(cov_matrix.columns)
cov_matrix.dropna(inplace=True)
cov_matrix= np.array(cov_matrix.values)
cov_matrix= cov_matrix.T
print(cov_matrix.shape)


# Calculate the expected returns
expected_returns = data['Adj Close'].pct_change().mean()
print("Expected Returns:")
print(expected_returns)
# print(data.shape[0],data.shape[1])
mark= Markowitz(cov_matrix.shape[0],cov_matrix.shape[1], cov_matrix)
optimal_weights, returns, risk= mark.optimal_portfolio()
print(optimal_weights)