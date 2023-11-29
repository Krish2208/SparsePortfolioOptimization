import numpy as np
from cvxopt import matrix, solvers

def solve_regularized_qp(Q, l1, l2, expected_return, target_return):
    n = Q.shape[0]
    # Define the objective function
    P = matrix((Q+l2 * np.eye(n)))
    q = matrix(np.ones(n) * l1)
    # Return constraint Gx <= h
    G_return = matrix(np.vstack((-np.array(expected_return), -np.eye(n))))
    h_return = matrix(np.hstack((-target_return, np.zeros(n))))
    

    # Combine the constraints
    G = G_return
    h = h_return
    # Equality constraint sum(x) = 1
    A = matrix(np.ones((1, n)))
    b = matrix(1.0)

    # Solve the QP problem
    sol = solvers.qp(P, q, G, h, A, b)
    #set up solver parameters
    solvers.options['kktreg'] = 1e-8
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10
    
    # Extract the solution (portfolio weights)
    
    x = np.array(sol['x']).flatten()
    return x