from scipy.optimize import minimize
import numpy as np

def l_1(model, input, label, epsilon):

    constraints = ({'type' : 'ineq',
             'fun' : lambda x: epsilon-np.sum(np.mod(x)),
             'jac' : lambda x: -np.sign(x)})

    bounds = tuple((-epsilon, epsilon) for y in input)

    res = minimize(lambda x: label*model(x), x0=input, method='SLSQP', bounds=bounds
                            , constraints=constraints, options={'maxiter': 10})

    return res.x