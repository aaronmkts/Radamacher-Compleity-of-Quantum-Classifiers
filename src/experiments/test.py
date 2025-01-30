import sys
import os

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..",
    )
)

# from codebase.datasets import gen_data
from codebase.train import train
from codebase.model import get_classifier
from codebase.adversary import get_attack

# Default Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import functools
import operator

# seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# Custom loss function
def loss_func(expvals, labels, alpha=10):
    loss = torch.mean(1 / (1 + torch.exp(alpha * labels * expvals)))
    return loss


# Simple classifier model
class QuantumAngleClassifier(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super(QuantumAngleClassifier, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Define the quantum device
        self.device = qml.device("default.qubit", wires=num_qubits)

        paulis = [qml.PauliZ(i) for i in range(self.num_qubits)]
        self.observable = functools.reduce(operator.matmul, paulis)

        # Define the quantum circuit
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            # Apply angle embedding
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            # Apply variational layers
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))

            # Measure expectation of Z on the first qubit
            return qml.expval(qml.PauliZ(num_qubits // 2))

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



class QuantumAmplClassifier(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super(QuantumAmplClassifier, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Define the quantum device
        self.device = qml.device("default.qubit", wires=num_qubits)

        paulis = [qml.PauliZ(i) for i in range(self.num_qubits)]
        self.observable = functools.reduce(operator.matmul, paulis)

        # Define the quantum circuit
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            # Apply amplitude embedding
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True, pad_with=0.)
            # Apply variational layers
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))

            # Measure expectation of Z on the first qubit
            return qml.expval(qml.PauliZ(num_qubits // 2))

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
        mu = np.pi * np.concatenate((np.ones(int(dimensions // 2)),
                                     (-np.ones(int(dimensions - dimensions // 2))) ** y[i])) / 4

        # Covariance is identity matrix
        cov = np.pi * np.eye(dimensions)/4

        # Generate data point
        x[i, :] = np.random.multivariate_normal(mu, cov)

    # Convert to torch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    return x, y


# Training function
def train_adv_model(classifier, train_loader, test_loader, epochs, learning_rate, device, attacker, epsilon):
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    attacker_cls = get_attack(attacker)
    train_losses = []


    for epoch in range(epochs):
        classifier.train()
        train_loss_epoch = 0.0

        #training
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

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {train_loss_epoch:.4f}")

    classifier.eval()

    #evaluation
    train_loss = 0.
    adv_train_loss = 0.
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_adv = attacker_cls(classifier, inputs, labels, loss_func, epsilon)
        labels = torch.where(labels == 0, 1, -1).float()  # Map labels {0, 1} -> {1, -1}

        #
        outputs = classifier(inputs).view(-1)
        outputs_adv = classifier(inputs_adv).view(-1)

        # Compute custom loss
        loss = loss_func(outputs, labels)
        train_loss += loss.item()

        loss = loss_func(outputs_adv, labels)
        adv_train_loss += loss.item()

    train_loss /= len(train_loader)
    adv_train_loss /= len(train_loader)

    adv_test_loss = 0.
    test_loss = 0.

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        inputs_adv = attacker_cls(classifier, inputs, labels, loss_func, epsilon)

        outputs_adv = classifier(inputs_adv).view(-1).detach()
        outputs = classifier(inputs).view(-1).detach()
        labels = torch.where(labels == 0, 1, -1).float()

        # Compute custom loss
        loss = loss_func(outputs, labels)
        test_loss += loss.item()

        loss = loss_func(outputs_adv, labels)
        adv_test_loss += loss.item()

    test_loss /= len(test_loader)
    adv_test_loss /= len(test_loader)

    return adv_train_loss, train_loss, adv_test_loss, test_loss


def train_model(classifier, train_loader, test_loader, threshold, learning_rate, device, attacker, epsilon):
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    attacker_cls = get_attack(attacker)
    train_losses = []

    train_loss_epoch=1.
    epoch = 0
    while (train_loss_epoch>threshold):
        epoch += 1
        classifier.train()
        train_loss_epoch = 0.0

        # training
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

        print(f"Epoch [{epoch + 1}] - Loss: {train_loss_epoch:.4f}")

    classifier.eval()

    # evaluation
    train_loss = 0.
    adv_train_loss = 0.
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_adv = attacker_cls(classifier, inputs, labels, loss_func, epsilon)
        labels = torch.where(labels == 0, 1, -1).float()  # Map labels {0, 1} -> {1, -1}

        #
        outputs = classifier(inputs).view(-1)
        outputs_adv = classifier(inputs_adv).view(-1)

        # Compute custom loss
        loss = loss_func(outputs, labels)
        train_loss += loss.item()

        loss = loss_func(outputs_adv, labels)
        adv_train_loss += loss.item()

    train_loss /= len(train_loader)
    adv_train_loss /= len(train_loader)

    adv_test_loss = 0.
    test_loss = 0.

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        inputs_adv = attacker_cls(classifier, inputs, labels, loss_func, epsilon)

        outputs_adv = classifier(inputs_adv).view(-1).detach()
        outputs = classifier(inputs).view(-1).detach()
        labels = torch.where(labels == 0, 1, -1).float()

        # Compute custom loss
        loss = loss_func(outputs, labels)
        test_loss += loss.item()

        loss = loss_func(outputs_adv, labels)
        adv_test_loss += loss.item()

    test_loss /= len(test_loader)
    adv_test_loss /= len(test_loader)

    return adv_train_loss, train_loss, adv_test_loss, test_loss


# Evaluation function
def evaluate_model(classifier, test_loader, device, epsilon, attacker):
    classifier.eval()
    adv_losses = []
    losses = []

    attacker_cls = get_attack(attacker)
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        inputs_adv = attacker_cls(classifier, inputs, labels, loss_func, epsilon)

        outputs = classifier(inputs_adv).view(-1).detach()
        labels = torch.where(labels == 0, 1, -1).float()
        adv_losses.append(torch.sum(loss_func(outputs, labels)))

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = classifier(inputs).view(-1).detach()
        labels = torch.where(labels == 0, 1, -1).float()
        losses.append(torch.sum(loss_func(outputs, labels)))

    return np.sum(adv_losses) / 1000., np.sum(adv_losses) / 1000.



# Main function
def main():
    # Parameters
    #num_train_samples = 10
    num_test_samples = 1000
    batch_size = 10
    epochs = 30
    learning_rate = 0.01
    num_layers = 4
    attacker = "l_2"
    epsilon = 0.4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runs = 20
    threshold=0.3
    dimensions = 2

    # control variables
    #dimensions_list = [2, 4, 6, 8, 10, 12, 14, 16]
    #deez = len(dimensions_list)

    training_data_list = np.linspace(10, 101, 10, dtype=int)
    deez = len(training_data_list)
    embeddings_list = ['angle', 'amplitude']

    angle = np.zeros((runs, deez, 4))
    ampl = np.zeros((runs, deez, 4))


    for num_train_samples in training_data_list:
        print(f"\nProcessing number of data: {num_train_samples}")

        # List to store loss differences for multiple runs
        loss_differences_runs = []
        x = np.zeros((runs, num_train_samples, dimensions))
        y = np.zeros((runs, num_train_samples))
        for run in range(runs):
            # Generate data
            x_train, y_train = gen_data(num_train_samples, dimensions)
            x[run] = x_train
            y[run] = y_train

            x_test, y_test = gen_data(num_test_samples, dimensions)

            for name in embeddings_list:
                print(f"\nProcessing embedding: {name}")

                print(f"Run {run + 1}/{runs}")

                train_dataset = TensorDataset(x_train, y_train)
                test_dataset = TensorDataset(x_test, y_test)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                if name == 'angle':
                    num_qubits = dimensions
                    classifier = QuantumAngleClassifier(num_qubits=num_qubits, num_layers=num_layers)
                else:
                    num_qubits = int(np.ceil(np.log2(dimensions)))
                    classifier = QuantumAmplClassifier(num_qubits=num_qubits, num_layers=num_layers)

                """
                # Training
                adv_train, train, adv_test, test = train_model(classifier, train_loader, test_loader,
                                                        threshold, learning_rate, device,
                                                        attacker, epsilon)

                # Adversarial Training
                """
                adv_train, train, adv_test, test = train_adv_model(classifier, train_loader, test_loader,
                                                               epochs, learning_rate, device,
                                                               attacker, epsilon)


                losses = np.array([adv_train, train, adv_test, test])

                i = int(num_train_samples//10-1)
                #i = int(np.log2(dimensions)-1)
                if name == 'angle':
                    angle[run][i] = losses
                else:
                    ampl[run][i] = losses

        #np.save(f"x_{dimensions}", x)
        #np.save(f"y_{dimensions}", y)


    np.save("angle_losses_num", angle)
    np.save("ampl_losses_num", ampl)

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