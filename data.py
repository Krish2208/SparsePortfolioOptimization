from markowitz import Markowitz
from regularised_markowitz import RegularisedMarkowitz
from regularised_markowitz_cvx import solve_regularized_qp
import yfinance as yf
import pandas as pd
import os
import pickle
import numpy as np

benchmark_index = False

# Define the tickers and time period
tickers = ["SO", "KMB", "K", "AZO", "GIS", "HSY",
           "PEP", "WMT", "NEM", "MCD", "AAPL", "CPB",
           "PG", "CAG", "FOH", "PPL", "DUK", "SYY",
           "REGN", "ABT", "JNJ", "ORLY", "MNST", "SHW",
           "THC", "GILD", "PG", "LLY", "BIIB", "DVA",
              "HCA", "HUM", "CI", "UNH", "WBA",
                "CVS", "AET", "AMGN", "BMY", "MRK",
                "PFE", "JPM", "BAC", "WFC", "C", "MS",
                "GS", "USB", "AXP", "COF", "DFS", "CB",
                "MET", "PRU", "ALL", "AIG", "AFL", "TRV",
                "MMC", "HIG", "CINF", "HII", "LMT", "GD", "COL", "HON", "GE", "MMM",
                "CAT", "DE", "PH", "ITW", "EMR", "ETN",
                "DOV", "FLS", "ROP", "SWK", "PNR",
                "AME", "XYL", "NOC", "GD", "LMT",
                  "TXT", "BA", "TDG"]

start_date = "2007-12-31"
end_date = "2012-12-31"

if os.path.exists("data.pkl"):
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
else:
    data = yf.download(tickers, start=start_date, end=end_date)
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)

if benchmark_index:
    # Get the data for the index (S&P 500)
    index_ticker = "^GSPC"
    index_data = yf.download(index_ticker, start=start_date, end=end_date)

    # Calculate the target return from the index price
    target_return = index_data['Adj Close'].pct_change().mean()
    print("Target Return:")
    print(target_return)
    
# Daily returns
daily_returns = data['Adj Close'].pct_change().apply(lambda x: pd.Series(x).fillna(x.mean()))

# Calculate the covariance matrix
cov_matrix = daily_returns.cov()
# print("Covariance Matrix:")
# print(cov_matrix)

# Calculate the expected returns
expected_returns = daily_returns.mean()
# print("Expected Returns:")
# print(expected_returns)
# print(expected_returns[expected_returns.isna()])


# Create an instance of the Markowitz class
# markowitz = Markowitz(expected_returns, cov_matrix)
# optimal_weights = markowitz.optimal_weights(0.05)
# print("Optimal Weights: ", optimal_weights)
# print("Mean Return: ", markowitz.portfolio_return(optimal_weights))
# print("Portfolio Variance: ", markowitz.portfolio_variance(optimal_weights))

# print(len(data['Adj Close']) - data['Adj Close'].isna().sum())
# total_returns = 0
# for i in range(len(optimal_weights)):
#     total_returns += optimal_weights[i]*expected_returns[i]*(len(data['Adj Close']) - data['Adj Close'].isna().sum())[i]
# print("Returns: ", total_returns)

# regularised_markowitz = RegularisedMarkowitz(expected_returns, cov_matrix, 5, 20)
# regularised_optimal_weights = regularised_markowitz.optimal_weights(0.05)
# print("Optimal Weights: ", regularised_optimal_weights)
# print("Mean Return: ", regularised_markowitz.portfolio_return(regularised_optimal_weights))
# print("Portfolio Variance: ", regularised_markowitz.portfolio_variance(optimal_weights))
# total_returns = 0
# for i in range(len(regularised_optimal_weights)):
#     total_returns += regularised_optimal_weights[i]*expected_returns[i]*(len(data['Adj Close']) - data['Adj Close'].isna().sum())[i]
# print("Returns: ", total_returns)
cov_matrix= np.array(cov_matrix)
expected_returns= np.array(expected_returns)
expected_returns= expected_returns.reshape((1,expected_returns.shape[0]))
# expected_returns= expected_returns.T
# print(expected_returns.size)
expected_returns= np.vstack([expected_returns, np.eye(expected_returns.shape[1])])
target_return= np.zeros(expected_returns.shape[1])
target_return= target_return.reshape((target_return.shape[0],1))
target_return= np.vstack([0.05, target_return])
print(target_return)
x= solve_regularized_qp(cov_matrix, 10, 1, expected_returns, target_return)
print("Optimal Weights: ", x)
