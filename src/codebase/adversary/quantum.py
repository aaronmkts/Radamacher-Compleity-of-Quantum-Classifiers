import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
import pennylane as qml
import torch
import torch.nn as nn



def quantum_attack(model, inputs, labels, loss_func, epsilon, num_layers=1):

    num_qubits = getattr(model, 'num_qubits')

    num_weights = 3*num_qubits
    w = torch.zeros((num_qubits, 3)).requires_grad_(True)

    bounds = tuple((-np.pi, np.pi) for i in range(num_weights))
    states = [model.circuit(inputs[i]) for i in range(len(labels))]

    adversarial_states = []
    for i in range(len(labels)):
        jac = lambda x: np.array(nd.Gradient(lambda y: -np.sum(np.abs(np.linalg.eigvals(model.circuit(inputs[i], adv_weights=np.reshape(y, (num_qubits, 3))) - states[i]))))(x))

        ineq_cons = {'type': 'ineq',
                     'fun': lambda x: epsilon - np.sum(np.abs(np.linalg.eigvals(model.circuit(inputs[i], adv_weights=np.reshape(x, (num_qubits, 3))) - states[i]))),
                     'jac': jac
                     }

        jac = lambda x: np.array(nd.Gradient(lambda y: labels[i] * model.forward(model.circuit(inputs[i], adv_weights=np.reshape(y, (num_qubits, 3)))).detach().numpy())(x))
        #jac = lambda x: qml.jacobian(labels[i]*model.forward_quantum([attacker.quantum_adversary(states[i], x)]))(x)

        res = minimize(lambda x: labels[i]*model.forward(model.circuit(inputs[i], adv_weights=np.reshape(x, (num_qubits, 3)))).detach().numpy(),
                   np.zeros(num_weights), jac=jac, method='SLSQP', constraints=(ineq_cons),
                   options={'disp': False, 'maxiter': 1}, bounds=bounds)

        adversarial_states.append(model.circuit(inputs[i], adv_weights=np.reshape(res.x, (num_qubits, 3))))

    return adversarial_states
