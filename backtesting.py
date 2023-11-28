from markowitz import Markowitz
from regularised_markowitz import RegularisedMarkowitz
from regularised_markowitz_half import RegularisedMarkowitzExperiment, RegularisedMarkowitzExperimentRiskAppetite
# from regularised_markowitz_cvx import solve_regularized_qp
import yfinance as yf
import pandas as pd
import os
import pickle
import numpy as np
import tqdm

# Define the tickers and time period
tickers = ["SO", "KMB", "K", "AZO", "GIS", "HSY",
           "PEP", "WMT", "NEM", "MCD", "AAPL", "CPB", "CAG", "PPL", "DUK", "SYY",
           "REGN", "ABT", "JNJ", "ORLY", "MNST", "SHW",
           "THC", "GILD", "PG", "LLY", "BIIB", "DVA", "HUM", "CI", "UNH", "WBA",
                "CVS", "AET", "AMGN", "BMY", "MRK",
                "PFE", "JPM", "BAC", "WFC", "C", "MS",
                "GS", "USB", "AXP", "COF", "DFS", "CB",
                "MET", "PRU", "ALL", "AIG", "AFL", "TRV",
                "MMC", "HIG", "CINF", "HON", "GE", "MMM",
                "CAT", "DE", "PH", "ITW", "EMR", "ETN",
                "DOV", "FLS", "ROP", "SWK", "PNR",
                "AME", "NOC", "GD", "LMT",
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
        
transaction_cost = 0
inital_wealth = 1000000
cumulative_wealth_std = inital_wealth
cumulative_wealth_reg = inital_wealth
cumulative_wealth_half = inital_wealth
cumulative_wealth_risk = inital_wealth

wealth_std = []
wealth_reg = []
wealth_half = []
wealth_risk = []

wealth_std.append(cumulative_wealth_std)
wealth_reg.append(cumulative_wealth_reg)
wealth_half.append(cumulative_wealth_half)
wealth_risk.append(cumulative_wealth_risk)

initial_weights = np.repeat(1/len(tickers), len(tickers))

for day in tqdm.tqdm(range(1, len(data))):
    daily_returns = data[:day]['Adj Close'].pct_change().apply(lambda x: pd.Series(x).fillna(x.mean()))
    
    # Calculate the covariance matrix
    cov_matrix = daily_returns.cov()
    
    # Calculate the expected returns
    expected_returns = daily_returns.mean()
    
    # Calculate the optimal weights
    standard_markowitz = Markowitz(expected_returns, cov_matrix)
    optimal_weights = standard_markowitz.optimal_weights(0.05)
    cumulative_wealth_std = cumulative_wealth_std*(1+np.dot(optimal_weights, daily_returns.iloc[-1]))
    wealth_std.append(cumulative_wealth_std)
    
    regularised_markowitz = RegularisedMarkowitz(expected_returns, cov_matrix, regularisation=0.1, regularisation2=5)
    optimal_weights = regularised_markowitz.optimal_weights(0.05)
    cumulative_wealth_reg = cumulative_wealth_reg*(1+np.dot(optimal_weights, daily_returns.iloc[-1]))
    wealth_reg.append(cumulative_wealth_reg)
    
    regularised_markowitz_half = RegularisedMarkowitzExperiment(expected_returns, cov_matrix, regularisation=5)
    optimal_weights = regularised_markowitz_half.optimal_weights(0.05)
    cumulative_wealth_half = cumulative_wealth_half*(1+np.dot(optimal_weights, daily_returns.iloc[-1]))
    wealth_half.append(cumulative_wealth_half)
    
    regularised_markowitz_risk = RegularisedMarkowitzExperimentRiskAppetite(expected_returns, cov_matrix, regularisation=5, risk_appetite=0.4)
    optimal_weights = regularised_markowitz_risk.optimal_weights(0.05)
    cumulative_wealth_risk = cumulative_wealth_risk*(1+np.dot(optimal_weights, daily_returns.iloc[-1]))
    wealth_risk.append(cumulative_wealth_risk)

print('Standard Markowitz')
print('---------------------------------')
print("Final Wealth: ", cumulative_wealth_std)
print("Returns: ", cumulative_wealth_std/inital_wealth)
print("Sharpe Ratio: ", (cumulative_wealth_std/inital_wealth - 1)/np.std(wealth_std))
print("Volatility: ", np.std(wealth_std))
print("Max Drawdown: ", np.max(np.maximum.accumulate(wealth_std) - wealth_std))
print("Min Drawdown: ", np.min(np.maximum.accumulate(wealth_std) - wealth_std))
print("Max Drawdown Duration: ", np.argmax(np.maximum.accumulate(wealth_std) - wealth_std))
print("Min Drawdown Duration: ", np.argmin(np.maximum.accumulate(wealth_std) - wealth_std))

print('\n\n\nl1 + l2 regularisation')
print('---------------------------------')
print("Final Wealth: ", cumulative_wealth_reg)
print("Returns: ", cumulative_wealth_reg/inital_wealth)
print("Sharpe Ratio: ", (cumulative_wealth_reg/inital_wealth - 1)/np.std(wealth_reg))
print("Volatility: ", np.std(wealth_reg))
print("Max Drawdown: ", np.max(np.maximum.accumulate(wealth_reg) - wealth_reg))
print("Min Drawdown: ", np.min(np.maximum.accumulate(wealth_reg) - wealth_reg))
print("Max Drawdown Duration: ", np.argmax(np.maximum.accumulate(wealth_reg) - wealth_reg))
print("Min Drawdown Duration: ", np.argmin(np.maximum.accumulate(wealth_reg) - wealth_reg))

print('\n\n\nl(1/2) regularisation')
print('---------------------------------')
print("Final Wealth: ", cumulative_wealth_half)
print("Returns: ", cumulative_wealth_half/inital_wealth)
print("Sharpe Ratio: ", (cumulative_wealth_half/inital_wealth - 1)/np.std(wealth_half))
print("Volatility: ", np.std(wealth_half))
print("Max Drawdown: ", np.max(np.maximum.accumulate(wealth_half) - wealth_half))
print("Min Drawdown: ", np.min(np.maximum.accumulate(wealth_half) - wealth_half))
print("Max Drawdown Duration: ", np.argmax(np.maximum.accumulate(wealth_half) - wealth_half))
print("Min Drawdown Duration: ", np.argmin(np.maximum.accumulate(wealth_half) - wealth_half))

print('\n\n\nl(1/2) regularisation with Risk Appetite')
print('---------------------------------')
print("Final Wealth: ", cumulative_wealth_risk)
print("Returns: ", cumulative_wealth_risk/inital_wealth)
print("Sharpe Ratio: ", (cumulative_wealth_risk/inital_wealth - 1)/np.std(wealth_risk))
print("Volatility: ", np.std(wealth_risk))
print("Max Drawdown: ", np.max(np.maximum.accumulate(wealth_risk) - wealth_risk))
print("Min Drawdown: ", np.min(np.maximum.accumulate(wealth_risk) - wealth_risk))
print("Max Drawdown Duration: ", np.argmax(np.maximum.accumulate(wealth_risk) - wealth_risk))
print("Min Drawdown Duration: ", np.argmin(np.maximum.accumulate(wealth_risk) - wealth_risk))


with open("wealth_std.pkl", "wb") as f:
    pickle.dump(wealth_std, f)
with open("wealth_reg.pkl", "wb") as f:
    pickle.dump(wealth_reg, f)
with open("wealth_half.pkl", "wb") as f:
    pickle.dump(wealth_half, f)
with open("wealth_risk.pkl", "wb") as f:
    pickle.dump(wealth_risk, f)
    
import matplotlib.pyplot as plt
plt.plot(wealth_std, label='Standard Markowitz')
plt.plot(wealth_reg, label='l1 + l2 regularisation')
plt.plot(wealth_half, label='l(1/2) regularisation')
plt.plot(wealth_risk, label='l(1/2) regularisation with Risk Appetite')
plt.legend()
plt.show()