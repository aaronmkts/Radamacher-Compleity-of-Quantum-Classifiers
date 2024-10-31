import torch
from scipy.optimize import minimize
import numpy as np

def L1(model, inputs, label, epsilon):

    constraints = ({'type' : 'ineq',
             'fun' : lambda x: epsilon-np.sum(np.mod(x)),
             'jac' : lambda x: -np.sign(x)})

    bounds = tuple((-epsilon, epsilon) for _ in inputs)
    res = minimize(lambda x: label*model(x), x0=inputs, method='SLSQP', bounds=bounds
                            , constraints=constraints, options={'maxiter': 10})

    return res.x