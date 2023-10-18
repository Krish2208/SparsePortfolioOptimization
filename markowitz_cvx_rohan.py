import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
solvers.options['show_progress'] = False


np.random.seed(123)

class MarkowitzOptimization():
    def __init__(self,n_assets, n_obs, returns):
        self.n_assets = n_assets
        self.n_obs = n_obs
        self.returns = returns

    def optimal_portfolio(self):
        n = len(self.returns)
        returns = np.asmatrix(self.returns)
        
        N = 100
        mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
        
        # Convert to cvxopt matrices
        S = opt.matrix(np.cov(self.returns))
        pbar = opt.matrix(np.mean(self.returns, axis=1))
        
        # Create constraint matrices
        G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
        h = opt.matrix(0.0, (n ,1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
        
        # Calculate efficient frontier weights using quadratic programming
        portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                    for mu in mus]
        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
        ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
        return np.asarray(wt), returns, risks
        

def random_portfolio(returns):
        ''' 
        Returns the mean and standard deviation of returns for a random portfolio
        '''
        p = np.asmatrix(np.mean(returns, axis=1))
        w = np.asmatrix(rand_weights(returns.shape[0]))
        C = np.asmatrix(np.cov(returns))
        mu = w * p.T
        sigma = np.sqrt(w * C * w.T)
        # This recursion reduces outliers to keep plots pretty
        if sigma > 2:
            return random_portfolio(returns)
        return mu, sigma

def rand_weights(self):
        ''' Produces n random weights that sum to 1 '''
        k = np.random.rand(self.n_assets)
        return k / sum(k)

# n=4
# return_vec = np.random.randn(n,1000)
# # returns = np.random.randn(1000,n)
# Markowitz = MarkowitzOptimization(4,1000,return_vec)

# weights, returns, risk= Markowitz.optimal_portfolio()
# print("Weights", weights)
# print("Returns", returns)
# print("Risk", risk)