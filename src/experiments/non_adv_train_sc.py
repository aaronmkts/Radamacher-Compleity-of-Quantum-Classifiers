import sys
import os

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..",
    )
)

#from codebase.datasets import gen_data
from codebase.train import train
from codebase.model import get_classifier

#Default Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import functools
import operator
#seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Custom loss function
def loss_func(expvals, labels):
    loss = torch.mean(1 / (1 + torch.exp(labels * expvals)))
    return loss

# Simple classifier model
class QuantumClassifier(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super(QuantumClassifier, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Define the quantum device
        self.device = qml.device("default.qubit", wires=num_qubits)

        paulis = [qml.PauliZ(i) for i in range(self.num_qubits)]
        self.observable =  functools.reduce(operator.matmul, paulis)
        # Define the quantum circuit
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            # Apply angle embedding
            #for i in range(self.num_qubits):
                #qml.RY(inputs[i], wires=i)
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True)
            # Apply variational layers
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            
            # Measure expectation of Z on the first qubit
            return qml.expval(qml.PauliZ(num_qubits//2))

        self.quantum_circuit = quantum_circuit

        # Initialize the weights for the quantum circuit
        num_weights = qml.templates.StronglyEntanglingLayers.shape(self.num_layers, self.num_qubits)
        self.weights = nn.Parameter(torch.randn(num_weights))

    def forward(self, x):
        outputs = []
        for i in range(x.size(0)):  # Iterate over the batch
            output = self.quantum_circuit(x[i], self.weights)
            outputs.append(output)
        return torch.stack(outputs)

def gen_data(num_samples, dimensions):

    # Initialize data and labels
    x = np.zeros((num_samples, dimensions))
    y = np.random.randint(0, 2, num_samples)  # Random labels {0, 1}

    for i in range(num_samples):
        # Define the mean for each class
        mu = np.pi*np.concatenate((np.ones(int(dimensions//2)),
                                    (-np.ones(int(dimensions - dimensions//2))) ** y[i]))/4
        
        # Covariance is identity matrix
        cov = 0.1 * np.eye(dimensions)
        
        # Generate data point
        x[i, :] = np.random.multivariate_normal(mu, cov)

    # Convert to torch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    
    return x, y
# Training function
def train_model(classifier, train_loader, test_loader, epochs, learning_rate, device):
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        classifier.train()
        train_loss_epoch = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.where(labels == 0, 1, -1).float()  # Map labels {0, 1} -> {1, -1}

            optimizer.zero_grad()
            outputs = classifier(inputs).view(-1)

            # Compute custom loss
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)

        # Evaluate accuracy on the test set
        test_accuracy = evaluate_model(classifier, test_loader, device)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {train_loss_epoch:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    return train_losses, test_accuracies

# Evaluation function
def evaluate_model(classifier, test_loader, device):
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs).view(-1)

            predictions = torch.sign(outputs)  # Convert outputs to binary predictions
            labels = torch.where(labels == 0, 1, -1).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

# Plotting function
def plot_results(train_losses, test_accuracies):
    epochs = len(train_losses)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Parameters
    num_train_samples = 1000
    num_test_samples = 500
    batch_size = 25
    dimension = 4  # Separation between Gaussian peaks
    epochs = 35
    learning_rate = 0.01
    num_qubits = 2  # Data is 2D
    num_layers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate data
    x_train, y_train = gen_data(num_train_samples, dimension)
    x_test, y_test = gen_data(num_test_samples, dimension)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize classifier
    classifier = QuantumClassifier(num_qubits=num_qubits, num_layers=num_layers)

    # Train the model
    train_losses, test_accuracies = train_model(classifier, train_loader, test_loader, epochs, learning_rate, device)

    # Plot results
    plot_results(train_losses, test_accuracies)

if __name__ == '__main__':
    main()