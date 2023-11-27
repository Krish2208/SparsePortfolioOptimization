import numpy as np
from cvxopt import matrix, solvers

def solve_regularized_qp(Q, l1, l2, expected_return, target_return):
    n = Q.shape[0]
    # Define the objective function
    P = matrix(1/2*(Q+2*l2 * np.eye(n)))
    q = matrix(np.ones(n) * l1)
    # Return constraint
    # G_return = matrix(-np.array([np.ones(n)]))
    G_return= matrix(-expected_return)
    h_return = matrix(-target_return)
    # print(G_return)

    # Combine the constraints
    G = G_return
    h = h_return
    # print(G)
    # print(h)
    # Equality constraint sum(x) = 1
    A = matrix(np.ones((1, n)))
    b = matrix(1.0)

    # Solve the QP problem
    sol = solvers.qp(P, q, G, h, A, b)
    solvers.options['kktreg'] = 1e-8
    # solvers.options['show_progress'] = False
    # solvers.options['abstol'] = 1e-7

    # Extract the solution (portfolio weights)
    x = np.array(sol['x']).flatten()

    return x