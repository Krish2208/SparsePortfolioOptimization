from markowitz import Markowitz
import yfinance as yf
import pandas as pd
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
cov_matrix = data['Adj Close'].pct_change().apply(
    lambda x: pd.Series(x).fillna(x.mean())).cov()
print("Covariance Matrix:")
print(cov_matrix)

# Calculate the expected returns
expected_returns = data['Adj Close'].pct_change().mean()
print("Expected Returns:")
print(expected_returns)


# Create an instance of the Markowitz class
markowitz = Markowitz(expected_returns, cov_matrix)
optimal_weights = markowitz.optimal_weights(0.0010)
print("Optimal Weights: ", optimal_weights)
print("Mean Return: ", markowitz.portfolio_return(optimal_weights))
print("Portfolio Variance: ", markowitz.portfolio_variance(optimal_weights))
