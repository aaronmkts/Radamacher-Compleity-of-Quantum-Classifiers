import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
import pennylane as qml
import torch
import torch.nn as nn


class QuantumAdversary(nn.Module):
    def __init__(self, num_qubits):
        super(QuantumAdversary, self).__init__()
        self.num_qubits = num_qubits

        # Define the quantum device
        self.device = qml.device("default.mixed", wires=self.num_qubits)

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def quantum_adversary_circuit(states, weights):

            qml.QubitDensityMatrix(states, wires=range(self.num_qubits))

            for j in range(self.num_qubits):
                qml.Rot(weights[j][0], weights[j][1], weights[j][2], wires=j)

            return qml.density_matrix(range(self.num_qubits))

        self.quantum_adversary = quantum_adversary_circuit


def loss_arg(labels, outputs):
    return torch.mean(labels*outputs)


def quantum_attack(model, states, labels, loss_func, epsilon, alpha=0.1):

    num_qubits = getattr(model, 'num_qubits')
    attacker = QuantumAdversary(num_qubits=num_qubits)
    weights = torch.zeros((len(labels), num_qubits, 3)).requires_grad_(True)

    adv_states = [attacker.quantum_adversary(states[i], weights[i]) for i in range(len(labels))]

    outputs = model.forward_quantum(adv_states)
    model.zero_grad()
    loss = loss_arg(labels, outputs)
    loss.backward()

    with torch.no_grad():
        perturbations = -alpha*weights.grad.sign()

    adv_states = [attacker.quantum_adversary(states[i], perturbations[i]) for i in range(len(labels))]

    for i in range(len(labels)):
        while (epsilon - np.sum(np.abs(np.linalg.eigvals(adv_states[i]-states[i])))<0.):
            perturbations[i] /= 2

    adv_states = attacker.quantum_adversary(states, perturbations)

    return adv_states
