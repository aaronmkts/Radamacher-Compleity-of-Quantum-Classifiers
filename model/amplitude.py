import torch
import pennylane as qml
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class AmplitudeEmbeddingClassifier(torch.nn.Module):
    """
    Class for creating a quantum machine learning (classification) model
    based on the StronglyEntanglingLayers template using amplitude embedding
    without data reuploading.

    Args:
        input_dim (int): Dimension of the input samples (must be 2^num_qubits).
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
        self.device = qml.device("lightning.qubit", wires=self.num_qubits)

        # Shape of the weights for the variational circuit
        self.weights_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.num_layers, n_wires=self.num_qubits
        )

        # Define the quantum circuit
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            # Amplitude embedding of the inputs
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True)
            # Variational layers
            qml.StronglyEntanglingLayers(weights=weights, wires=range(self.num_qubits))
            # Measurement
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(self.num_qubits - 1))

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
        # Pass the input directly to the quantum circuit
        return self.qcircuit(x)
