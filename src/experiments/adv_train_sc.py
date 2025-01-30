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
from codebase.adversary import get_attack

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


# Custom loss function
def loss_func(expvals, labels, alpha=10):
    loss = torch.mean(1 / (1 + torch.exp(alpha*labels * expvals)))
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
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            #qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True)
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
def train_adv_model(classifier, train_loader, test_loader, epochs, learning_rate, device, attacker, epsilon, compare):
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    attacker_cls = get_attack(attacker)
    train_losses = []
    test_accuracies = []

    if compare:
        train_loss_epoch = 1.
        epoch_num=1
        while (train_loss_epoch>0.251 and epoch_num<=15):
            epoch_num +=1
            classifier.train()
            train_loss_epoch = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = torch.where(labels == 0, 1, -1).float()  # Map labels {0, 1} -> {1, -1}

                # Generate adversarial examples
                inputs_adv = attacker_cls(classifier, inputs, labels, loss_func, epsilon)

                optimizer.zero_grad()
                outputs = classifier(inputs_adv).view(-1)

                # Compute custom loss
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()

            train_loss_epoch /= len(train_loader)
            train_losses.append(train_loss_epoch)

        # Evaluate accuracy on the test set
        test_accuracy, test_loss = evaluate_model(classifier, test_loader, device, epsilon, attacker)

        if epoch_num<=15:
            return train_losses[-1], test_loss

        return -1., -1.
    else:
        for epoch in range(epochs):
            classifier.train()
            train_loss_epoch = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = torch.where(labels == 0, 1, -1).float()  # Map labels {0, 1} -> {1, -1}

                # Generate adversarial examples
                inputs_adv = attacker_cls(classifier, inputs, labels, loss_func, epsilon)

                optimizer.zero_grad()
                outputs = classifier(inputs_adv).view(-1)

                # Compute custom loss
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()

            train_loss_epoch /= len(train_loader)
            train_losses.append(train_loss_epoch)

            # Evaluate accuracy on the test set
            test_accuracy, test_loss = evaluate_model(classifier, test_loader, device, epsilon, attacker)
            test_accuracies.append(test_accuracy)

            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {train_loss_epoch:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        return train_losses, test_accuracies

# Evaluation function
def evaluate_model(classifier, test_loader, device, epsilon, attacker):
    classifier.eval()
    correct = 0
    total = 0
    losses = []

    attacker_cls = get_attack(attacker)
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        
        inputs_adv = attacker_cls(classifier, inputs, labels, loss_func, epsilon)

        outputs = classifier(inputs_adv).view(-1).detach()
        labels = torch.where(labels == 0, 1, -1).float()
        losses.append(torch.sum(loss_func(outputs, labels)))


        predictions = torch.sign(outputs)  # Convert outputs to binary predictions
        correct += (predictions == labels).sum().item()
        total += labels.size(0)


    accuracy = 100 * correct / total
    return accuracy, np.sum(losses)/500.
"""
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
"""
# Main function
def main():
    # Parameters
    num_train_samples = 1000
    num_test_samples = 500
    batch_size = 25
    #dimension = 4  # Separation between Gaussian peaks
    epochs = 35
    learning_rate = 0.01
    #num_qubits = 16  # Data is 2D
    num_layers = 2
    attacker = "l_2"
    epsilon = 0.05
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # control variables
    dimensions_list = [2, 4, 8, 16]
    embeddings_list = ['angle', 'amplitude']
    loss_differences_dict = {'angle': [], 'amplitude': []}

    plt.figure()

    for name in embeddings_list:
        print(f"\nProcessing embedding: {name}")
        loss_differences = []

        runs = 5

        for dimensions in dimensions_list:
            print(f"\nProcessing dimension: {dimensions}")

            # List to store loss differences for multiple runs
            loss_differences_runs = []

            for run in range(runs):
                print(f"Run {run + 1}/{runs}")

                # Generate data
                x_train, y_train = gen_data(num_train_samples, dimensions)
                x_test, y_test = gen_data(num_test_samples, dimensions)

                train_dataset = TensorDataset(x_train, y_train)
                test_dataset = TensorDataset(x_test, y_test)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                if name == 'angle':
                    num_qubits = dimensions
                else:
                    num_qubits = int(np.ceil(np.log2(dimensions)))

                # Initialize classifier
                classifier = QuantumClassifier(num_qubits=num_qubits, num_layers=num_layers)

                # Train the model
                train_loss, test_loss = train_adv_model(classifier, train_loader, test_loader,
                                                                epochs, learning_rate, device,
                                                                attacker, epsilon, True)

                if train_loss > -1.:
                    loss_differences_runs.append(np.absolute(test_loss - train_loss))


            # max of loss differences over runs indicates worst case generalization
            loss_differences.append(np.mean(loss_differences_runs))

            # After processing all dimensions for this embedding, store the loss_differences
        loss_differences_dict[name] = loss_differences

        plt.plot(dimensions_list, loss_differences_dict[name], marker='o', label=name)

    plt.xlabel('Dimensions (d)')
    plt.ylabel('Test Loss (Max over 5 runs)')
    plt.title('Max Modulus of Loss Difference vs Dimensions')
    plt.grid(True)
    plt.legend()
    plt.show()



    """
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
    train_losses, test_accuracies = train_adv_model(classifier, train_loader, test_loader, 
                                                    epochs, learning_rate, device,
                                                    attacker, epsilon)

    # Plot results
    plot_results(train_losses, test_accuracies)
    """

if __name__ == '__main__':
    main()