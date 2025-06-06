from scipy.optimize import minimize
import numpy as np
import torch
from scipy.optimize import minimize
import numpy as np


def L1(model, inputs, labels, loss_func, epsilon):

    inputs_adv = inputs.clone().detach().requires_grad_(False).numpy()
    labels_adv = labels.clone().detach().numpy()

    constraints = ({'type' : 'ineq',
             'fun' : lambda x: epsilon-np.sum(np.abs(x)),
             'jac' : lambda x: -np.sign(x)})

    adv_inputs = []

    for i in range(len(inputs_adv)):
        bounds = tuple((-epsilon, epsilon) for y in inputs_adv[i])

        res = minimize(lambda x: labels_adv[i]*model(torch.from_numpy(x)).detach().numpy(), x0=inputs_adv[i]
                       , method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 10})

        adv_inputs.append(res.x)

        ret = np.array(adv_inputs)

    return torch.from_numpy(ret)

