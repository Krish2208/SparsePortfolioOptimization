from markowitz import Markowitz
from regularised_markowitz import RegularisedMarkowitz
from regularised_markowitz_half import RegularisedMarkowitzExperiment, RegularisedMarkowitzExperimentRiskAppetite
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



# Create an instance of the Markowitz class

print('\n\n\nExperiment Standard Markowitz')
print('---------------------------------')
markowitz = Markowitz(expected_returns, cov_matrix)
optimal_weights = markowitz.optimal_weights(0.05)
print("Optimal Weights: ", optimal_weights)
print("Mean Return: ", markowitz.portfolio_return(optimal_weights))
print("Portfolio Variance: ", markowitz.portfolio_variance(optimal_weights))

print('\n\n\nExperiment l1 + l2 regularisation')
print('---------------------------------')
regularised_markowitz = RegularisedMarkowitz(expected_returns, cov_matrix, 5, 5)
regularised_optimal_weights = regularised_markowitz.optimal_weights(0.01)
print("Optimal Weights: ", regularised_optimal_weights)
print("Mean Return: ", regularised_markowitz.portfolio_return(regularised_optimal_weights))
print("Portfolio Variance: ", regularised_markowitz.portfolio_variance(optimal_weights))

print('\n\n\nExperiment l(1/2) regularisation')
print('---------------------------------')
regularised_markowitz_experiment = RegularisedMarkowitzExperiment(expected_returns, cov_matrix, 5)
regularised_optimal_weights_experiment = regularised_markowitz_experiment.optimal_weights(0.05)
print("Optimal Weights: ", regularised_optimal_weights_experiment)
print("Sum of Weights: ", np.sum(regularised_optimal_weights_experiment))
print("Mean Return: ", regularised_markowitz_experiment.portfolio_return(regularised_optimal_weights_experiment))
print("Portfolio Variance: ", regularised_markowitz_experiment.portfolio_variance(optimal_weights))


print('\n\n\nExperiment l(1/2) regularisation with Risk Appetite')
print('---------------------------------')
regularised_markowitz_experiment_risk_appetite = RegularisedMarkowitzExperimentRiskAppetite(expected_returns, cov_matrix, 5, 0.2)
regularised_optimal_weights_experiment_risk_appetite = regularised_markowitz_experiment_risk_appetite.optimal_weights(0.05)
print("Optimal Weights: ", regularised_optimal_weights_experiment_risk_appetite)
print("Sum of Weights: ", np.sum(regularised_optimal_weights_experiment_risk_appetite))
print("Mean Return: ", regularised_markowitz_experiment_risk_appetite.portfolio_return(regularised_optimal_weights_experiment_risk_appetite))
print("Portfolio Variance: ", regularised_markowitz_experiment_risk_appetite.portfolio_variance(optimal_weights))

print('\n\n\nExperiment l1 l2 using cvxopt')
print('---------------------------------')
cov_matrix= np.array(cov_matrix)
expected_returns= np.array(expected_returns)
target_return = 0.15
expected_returns /= np.std(expected_returns)
cov_matrix /= np.outer(np.std(expected_returns), np.std(expected_returns))
x= solve_regularized_qp(cov_matrix, 0.1, 0.1, expected_returns, target_return)
print("Optimal Weights: ", x)