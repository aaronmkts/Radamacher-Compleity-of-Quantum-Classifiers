#### Hyperparameters ####

import torch
import pennylane as qml
import numpy as np

#### Hyperparameters ####
input_dim = 256     # Dimension of the input samples
num_classes = 4     # Number of output classes
num_layers = 32     # Number of layers in the variational circuit
num_qubits = 8      # Number of qubits in the circuit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class AngleEmbeddingClassifier(torch.nn.Module):
    """
    Class for creating a quantum machine learning (classification) model
    based on the StronglyEntanglingLayers template using angle embedding
    without data reuploading.

    Args:
        input_dim (int): Dimension of the input samples.
        output_dim (int): Number of output classes.
        num_qubits (int): Number of qubits in the circuit.
        num_layers (int): Number of layers within the StronglyEntanglingLayers template.
    """
    def __init__(self, input_dim, output_dim, num_qubits, num_layers):
        super().__init__()
        torch.manual_seed(1337)  # Fixed seed for reproducibility
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Classical preprocessing layer to reduce input_dim to num_qubits
        self.pre_net = torch.nn.Linear(self.input_dim, self.num_qubits)

        # Quantum device setup
        self.device = qml.device("lightning.qubit", wires=self.num_qubits)

        # Shape of the weights for the variational circuit
        self.weights_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.num_layers, n_wires=self.num_qubits
        )

        # Define the quantum circuit
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            # Angle embedding of the inputs
            qml.AngleEmbedding(features=inputs, wires=range(self.num_qubits))
            # Variational layers
            qml.StronglyEntanglingLayers(weights=weights, wires=range(self.num_qubits))
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

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
        # Pass input through the classical preprocessing layer
        x = self.pre_net(x)
        # Pass the processed input to the quantum circuit
        return self.qcircuit(x)
