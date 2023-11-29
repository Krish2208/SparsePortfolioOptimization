import numpy as np
from scipy.optimize import minimize

class Markowitz:
    
    """
    Class to perform Markowitz portfolio optimization
    returns: expected returns of the assets
    covariance: covariance matrix of the assets
    target_return: target return for the portfolio
    """
    
    def __init__(self, returns, covariance):
        self.returns = returns
        self.covariance = covariance
        self.n = covariance.shape[0]
        self.target_return = None
    
    
    # Define function to calculate portfolio variance given weights
    # weights: weights of the assets in the portfolio
    def portfolio_variance(self, weights):
        return np.dot(weights.T, np.dot(self.covariance, weights))
    
    
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