import numpy as np
from cvxopt import matrix, solvers

def solve_regularized_qp(Q, l1, l2, target_return):
    n = Q.shape[0]

    P = matrix(Q + l2 * np.eye(n))
    q = matrix(np.ones(n) * l1)
    
    # Return constraint
    G_return = matrix(-np.array([np.ones(n)]))
    h_return = matrix(-target_return)

    # Combine the constraints
    G = G_return
    h = h_return

    # Equality constraint sum(x) = 1
    A = matrix(np.ones((1, n)))
    b = matrix(1.0)

    # Solve the QP problem
    sol = solvers.qp(P, q, G, h, A, b)

    # Extract the solution (portfolio weights)
    x = np.array(sol['x']).flatten()

    return x