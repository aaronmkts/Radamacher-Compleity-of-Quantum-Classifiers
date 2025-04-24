import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
import pennylane as qml
import torch
import torch.nn as nn


def loss_arg(labels, outputs):
    return torch.mean(labels*outputs)


def quantum_attack(model, l_min, inputs, labels, loss_func, epsilon, alpha=0.3, max_count=10,steps=3):

    model.zero_grad()
    num_qubits = getattr(model, 'num_qubits')
    state_dim = 2**num_qubits

    w = torch.zeros((len(labels), num_qubits, 3)).requires_grad_(True)
    states = [(1 - l_min * state_dim) * model.embedding(x).to(torch.complex64) + l_min * torch.eye(state_dim, dtype=torch.complex64) for x in
              inputs]

    forw_states = [model.circuit(states[i], adv_weights=w[i]).to(torch.complex64) for i in range(len(labels))]

    outputs = model.forward(forw_states)

    loss = loss_arg(labels, outputs)
    loss.backward()

    with torch.no_grad():
        perturbations = -alpha*w.grad.sign()

    adv_states = [model.circuit(states[i], adv_weights=perturbations[i], full_forw=False).to(torch.complex64) for i in range(len(labels))]

    for i in range(len(labels)):
        count = 0
        while (epsilon - torch.max(torch.abs(torch.linalg.eigvals(adv_states[i]-states[i])))<0. and count<max_count):
            count += 1
            perturbations[i] /= 2.
            adv_states[i] = model.circuit(states[i], adv_weights=perturbations[i], full_forw=False).to(torch.complex64)

        if count == max_count:
            perturbations[i] = torch.zeros((num_qubits, 3))
            #adv_states[i] = model.circuit(states[i], adv_weights=perturbations[i], full_forw=False).to(torch.complex64)

        #print(perturbations)
    adv_states = [model.circuit(states[i], adv_weights=perturbations[i]) for i in range(len(labels))]

    return torch.stack(adv_states)