import numpy as np
from scipy.optimize import minimize

# # Define assets and their expected returns
# assets = ['AAPL', 'GOOG', 'IBM', 'XOM']
# returns = np.array([0.1, 0.2, 0.15, 0.05])

# # Define covariance matrix for the assets
# covariance = np.array([[0.01, 0.005, 0.004, 0.002],
#                        [0.005, 0.02, 0.007, 0.003],
#                        [0.004, 0.007, 0.015, 0.001],
#                        [0.002, 0.003, 0.001, 0.01]])

# # Define target return for the portfolio
# target_return = 0.12

# # Define function to calculate portfolio variance given weights
# def portfolio_variance(weights):
#     return 0.5*np.dot(weights.T, np.dot(covariance, weights))

# # Define function to calculate portfolio return given weights
# def portfolio_return(weights):
#     return np.dot(weights.T, returns)

# # Define function to minimize portfolio variance subject to target return constraint
# def optimize_portfolio(initial_weights):
#     constraints = ({'type': 'ineq', 'fun': lambda x: portfolio_return(x) - target_return},
#                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#     bounds = tuple((0, 1) for asset in assets)
#     result = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
#     return result.x

# # Call optimization function with initial weights to obtain optimal weights for the portfolio
# initial_weights = np.array([0.25, 0.25, 0.25, 0.25])
# optimal_weights = optimize_portfolio(initial_weights)
# print(optimal_weights)


class RegularisedMarkowitz:
    
    """
    Class to perform Markowitz portfolio optimization
    returns: expected returns of the assets
    covariance: covariance matrix of the assets
    target_return: target return for the portfolio
    """
    
    def __init__(self, returns, covariance, regularisation=0, regularisation2=0):
        self.returns = returns
        self.covariance = covariance
        self.n = covariance.shape[0]
        self.target_return = None
        self.regularisation = regularisation
        self.regularisation2 = regularisation2
    
    
    # Define function to calculate portfolio variance given weights
    # weights: weights of the assets in the portfolio
    def portfolio_variance(self, weights):
        return np.dot(weights.T, np.dot(self.covariance, weights)) + self.regularisation*np.sum(np.abs(weights)) + self.regularisation2*np.sum(np.square(weights))
    
    
    # Define function to calculate portfolio return given weights
    # weights: weights of the assets in the portfolio
    def portfolio_return(self, weights):
        return np.dot(weights.T, self.returns)
    
    
    # Define function to minimize portfolio variance subject to target return constraint
    # initial_weights: initial weights of the assets in the portfolio
    def optimize_portfolio(self, initial_weights):
        constraints = ({'type': 'ineq', 'fun': lambda x: self.portfolio_return(x) - self.target_return},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(self.n))
        result = minimize(self.portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    
    
    # Define function to obtain optimal weights for the portfolio    
    # target_return: target return for the portfolio
    # initial_weights: initial weights of the assets in the portfolio, default is equal weights
    def optimal_weights(self, target_return, initial_weights = None):
        self.target_return = target_return
        if initial_weights is None:
            initial_weights = np.repeat(1/self.n, self.n)
        optimal_weights = self.optimize_portfolio(initial_weights)
        return optimal_weights