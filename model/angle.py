import torch
import pennylane as qml
import numpy as np
import functools
import operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class AngleEmbeddingClassifier(torch.nn.Module):
    """
    Class for creating a quantum machine learning (classification) model
    based on the StronglyEntanglingLayers template using angle embedding
    without data reuploading.

    Args:
        output_dim (int): Number of output classes.
        num_qubits (int): Number of qubits in the circuit.
        num_layers (int): Number of layers within the StronglyEntanglingLayers template.
    """
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        torch.manual_seed(1337)  # Fixed seed for reproducibility
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Quantum device setup
        self.device = qml.device("default.qubit", wires=self.num_qubits)

        # Shape of the weights for the variational circuit
        self.weights_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.num_layers, n_wires=self.num_qubits
        )

        paulis = [qml.PauliZ(i) for i in range(self.num_qubits)]
        self.observable =  functools.reduce(operator.matmul, paulis)
        # Define the quantum circuit
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            # Angle embedding of the inputs
            qml.AngleEmbedding(features=inputs, wires=range(self.num_qubits), rotation='Y')
            # Variational layers
            qml.StronglyEntanglingLayers(weights=weights, wires=range(self.num_qubits))
            # Measurement
            return qml.expval(self.observable)

        # Initialize the parameters for the quantum circuit
        param_shapes = {"weights": self.weights_shape}
        init_vals = {"weights": 0.1 * torch.rand(self.weights_shape)}

        # Create the quantum circuit as a TorchLayer
        self.qcircuit = qml.qnn.TorchLayer(
            qnode=circuit,
            weight_shapes=param_shapes,
            init_method=init_vals
        )

    def forward(self, x):
   
        return self.qcircuit(x)



