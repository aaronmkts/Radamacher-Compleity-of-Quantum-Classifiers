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
import scipy as sp
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
    loss = (1 / (1 + torch.exp(alpha * labels * expvals)))
    return loss

def risk(expvals, labels, alpha=10):
    loss = torch.mean((1 / (1 + torch.exp(alpha * labels * expvals))))
    return loss

def gen_data(num_samples, dimensions):
    # Initialize data and labels
    x = np.zeros((num_samples, dimensions))
    y = np.random.randint(0, 2, num_samples)  # Random labels {0, 1}

    for i in range(num_samples):
        # Define the mean for each class
        mu = np.pi * np.concatenate((np.ones(int(dimensions // 2)),
                                     (-np.ones(int(dimensions - dimensions // 2))) ** y[i])) / 4

        # Covariance is identity matrix
        cov = np.pi * np.eye(dimensions)/8

        # Generate data point
        x[i, :] = np.random.multivariate_normal(mu, cov)

    # Convert to torch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    return x, y


# Simple classifier model
class QuantumClassifier(nn.Module):
    def __init__(self, embedding, data_dim, num_layers):
        super(QuantumClassifier, self).__init__()
        self.embedding = embedding
        if self.embedding == 'angle':
            self.num_qubits = data_dim
        else:
            self.num_qubits = int(np.ceil(np.log2(data_dim)))

        self.num_layers = num_layers

        # Define the quantum device
        self.device = qml.device("default.mixed", wires=self.num_qubits)

        paulis = [qml.PauliZ(i) for i in range(self.num_qubits)]
        self.observable = functools.reduce(operator.matmul, paulis)
        num_weights = qml.templates.StronglyEntanglingLayers.shape(self.num_layers, self.num_qubits)
        self.weights = nn.Parameter(torch.randn(num_weights))

        # Define the quantum circuit
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def embedding(inputs):
            if self.embedding == 'angle':
                for i in range(self.num_qubits): # Apply angle embedding
                    qml.RY(inputs[i], wires=i)
            else:
                qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True, pad_with=0.)

            return qml.density_matrix(range(self.num_qubits))


        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(state, adv_weights=torch.zeros(self.num_qubits, 3), full_forw=True):
            qml.QubitDensityMatrix(state, wires=range(self.num_qubits))
            for j in range(self.num_qubits):
                qml.Rot(adv_weights[j][0], adv_weights[j][1], adv_weights[j][2], wires=j)

            if full_forw:
                qml.templates.StronglyEntanglingLayers(self.weights, wires=range(self.num_qubits))

            return qml.density_matrix(range(self.num_qubits))

        self.circuit = circuit
        self.embedding = embedding

        # Initialize the weights for the quantum circuit


    def forward(self, x):
        outputs = []
        for i in range(len(x)):  # Iterate over the batch
            #print(x[i].size())
            outputs.append(torch.trace(torch.matmul(x[i].to(torch.complex64), torch.tensor(self.observable.matrix(), dtype=torch.complex64))).to(torch.float64))

        return torch.stack(outputs)


def train_adv_model(model, l_min, train_loader, epochs, learning_rate, device, attacker, epsilon):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    attacker_cls = get_attack(attacker)
    train_losses = []

    model.train()
    for epoch in range(epochs):

        train_loss_epoch = 0.0

        # training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Generate adversarial examples
            adv_states = attacker_cls(model, l_min, inputs, labels, loss_func, epsilon)

            optimizer.zero_grad()
            outputs = model.forward(adv_states)

            # Compute custom loss
            loss = risk(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)

        if epoch == epochs - 1:
            print(f"Loss: {train_loss_epoch:.4f}")

    model.eval()

    return model

def evaluate(model, l_min, train_loader, test_loader, device, attacker, epsilon):

    attacker_cls = get_attack(attacker)
    num_qubits = getattr(model, "num_qubits")
    model.eval()
    state_dim = 2**num_qubits

    #evaluation
    train_loss = 0.
    adv_train_loss = 0.
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        states = [(1-l_min*state_dim)*model.embedding(x).to(torch.complex64)+l_min*torch.eye(state_dim, dtype=torch.complex64) for x in inputs]
        forw_states = [model.circuit(x).to(torch.complex64) for x in states]

        outputs = model.forward(forw_states)

        # Generate adversarial examples
        adv_states = attacker_cls(model, l_min, inputs, labels, loss_func, epsilon)
        outputs_adv = model.forward(adv_states)

        #print(outputs_adv-outputs)

        # Compute custom loss
        loss = risk(outputs, labels)
        train_loss += loss.item()

        loss = risk(outputs_adv, labels)
        adv_train_loss += loss.item()

    train_loss /= len(train_loader)
    adv_train_loss /= len(train_loader)

    adv_test_loss = 0.
    test_loss = 0.

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        states = [(1 - l_min * state_dim) * model.embedding(x).to(torch.complex64) + l_min * torch.eye(state_dim) for x
                  in inputs]
        forw_states = [model.circuit(x) for x in states]

        adv_states = attacker_cls(model, l_min, inputs, labels, loss_func, epsilon)

        outputs = model.forward(forw_states)
        outputs_adv = model.forward(adv_states)

        loss = risk(outputs, labels)
        test_loss += loss.item()

        loss = risk(outputs_adv, labels)
        adv_test_loss += loss.item()


    test_loss /= len(test_loader)
    adv_test_loss /= len(test_loader)

    return adv_train_loss, train_loss, adv_test_loss, test_loss

def main2():
    num_test_samples=200
    dimension = 6
    embedding = "amplitude"
    epsilon = 0.08
    lambda_min = 0.09
    num_layers = 4
    batch_size = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attacker = "QFGSM"

    x = torch.tensor(np.load("x_r=30_d=6_m=20.npy")[3])
    y = torch.tensor(np.load("y_r=30_d=6_m=20.npy")[3])
    y = torch.where(y == 0, 1, -1).float()

    x_test, y_test = gen_data(num_test_samples, dimension)
    y_test = torch.where(y_test == 0, 1, -1).float()

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #print(y)

    model = QuantumClassifier(embedding=embedding, data_dim=dimension, num_layers=num_layers)

    model = train_adv_model(model, 0., train_loader, 20, 0.1, device, attacker, epsilon)
    adv_train, train, adv_test, test = evaluate(model, lambda_min, train_loader, test_loader, device, attacker, epsilon)

    print("Adversarial",adv_test,adv_train)
    print("Standard", test,train)
    #plt.plot(train_losses_ep)
    #plt.show()



def main1():
    # Parameters

    num_train_samples = 20
    num_test_samples = 200
    batch_size = 10
    epochs = 30
    learning_rate = 0.1
    num_layers = 4
    attacker = "QFGSM"
    #epsilon = 0.04
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runs = 20
    embedding = 'amplitude'

    """ 
    # control variables
    dimensions_list = [2, 4, 8, 16]
    deez = len(dimensions_list)
    lambda_mins = [0., 0.5]


    noiseless = np.zeros((runs, deez, 4))
    noisy = np.zeros((runs, deez, 4))

    for dimensions in dimensions_list:
        print(f"\nProcessing dimension: {dimensions}")

        for run in range(runs):
            # Generate data
            print(f"Run {run + 1}/{runs}")
            #x_train, y_train = gen_data(num_train_samples, dimensions)
            x = np.load(f'x_r=30_d={dimensions}_m=20.npy')
            x_train = torch.tensor(x[run], dtype=torch.float64)
            y = np.load(f'y_r=30_d={dimensions}_m=20.npy')
            y_train = torch.tensor(y[run], dtype=torch.float64)
            y_train = torch.where(y_train == 0, 1, -1).float()  # Map labels {0, 1} -> {1, -1}
            #x[run] = x_train
            #y[run] = y_train

            x_test, y_test = gen_data(num_test_samples, dimensions)
            y_test = torch.where(y_test == 0, 1, -1).float()

            for lambda_min in lambda_mins:
                print(f"\nProcessing noise level: {lambda_min}")

                train_dataset = TensorDataset(x_train, y_train)
                test_dataset = TensorDataset(x_test, y_test)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


                classifier = QuantumClassifier(embedding=embedding, data_dim=dimensions, num_layers=num_layers, lambda_min=lambda_min)

                # Adversarial Training

                adv_train, train, adv_test, test = train_adv_model(classifier, train_loader, test_loader,
                                                               epochs, learning_rate, device,
                                                               attacker, epsilon)


                losses = np.array([adv_train, train, adv_test, test])

                ds = int(np.log2(dimensions)-1)

                if lambda_min == 0.:
                    noiseless[run][ds] = losses
                else:
                    noisy[run][ds] = losses

        #np.save(f"x_r=30_d={dimensions}_m=20", x)
        #np.save(f"y_r=30_d={dimensions}_m=20", y)


    np.save("noisy_losses_BIM", noisy)
    np.save("noiseless_losses_BIM", noiseless)
    
    dimensions = 8
    lambda_min = 0.0
    classifier = QuantumClassifier(embedding=embedding, data_dim=dimensions, num_layers=num_layers,
                                   lambda_min=lambda_min)

    attacker = get_attack('QFGSM')
    x = np.load(f'x_r=30_d={dimensions}_m=20.npy')
    y = np.load(f'y_r=30_d={dimensions}_m=20.npy')
    count = 0
    
    inputs = torch.tensor(x[0])
    labels = torch.tensor(y[0])
    labels = torch.where(labels == 0, 1, -1).float()

    states = [classifier.circuit(x) for x in inputs]

    outputs = classifier.forward(states)

    adversarial_states = attacker(classifier, inputs, labels, loss_func, epsilon)

    outputs_adv = classifier.forward(adversarial_states)

    losses = loss_func(outputs, labels).detach()

    adv_losses = loss_func(outputs_adv, labels).detach()

    print(adv_losses-losses)

    #print(adv_losses-losses)

    plt.plot(losses, color='green', label='Gen')
    plt.plot(adv_losses, color='red', label='Adv_Gen')
    plt.legend()
    plt.show()
    """

    # control variables
    dimensions = 6
    lambda_min = 0.011
    epsilons = np.linspace(0.001, lambda_min-0.001, 8)
    eez = len(epsilons)
    l_mins = [0.0, lambda_min]

    noiseless = np.zeros((runs, eez, 4))
    noisy = np.zeros((runs, eez, 4))


    for run in range(runs):
        print(f"Run {run + 1}/{runs}")
        classifier = QuantumClassifier(embedding=embedding, data_dim=dimensions, num_layers=num_layers)

        x_test, y_test = gen_data(num_test_samples, dimensions)
        y_test = torch.where(y_test == 0, 1, -1).float()
        x = np.load(f'x_r=30_d={dimensions}_m=20.npy')
        x_train = torch.tensor(x[run], dtype=torch.float64)
        y = np.load(f'y_r=30_d={dimensions}_m=20.npy')
        y_train = torch.tensor(y[run], dtype=torch.float64)
        y_train = torch.where(y_train == 0, 1, -1).float()  # Map labels {0, 1} -> {1, -1}
        for l_min in l_mins:
            print(f"\nProcessing noise level: {l_min}")

            for i in range(len(epsilons)):
                epsilon = epsilons[i]
                print(f"\nProcessing adversarial strength: {epsilon}")

                train_dataset = TensorDataset(x_train, y_train)
                test_dataset = TensorDataset(x_test, y_test)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # Adversarial Training

                if ((i == 0) and (l_min==0.)):
                    classifier = train_adv_model(classifier, 0., train_loader, epochs, learning_rate, device, attacker, epsilon)

                adv_train, train, adv_test, test = evaluate(classifier, l_min, train_loader, test_loader, device, attacker, epsilon)

                losses = np.array([adv_train, train, adv_test, test])


                if l_min == 0.:
                    noiseless[run][i] = losses
                else:
                    noisy[run][i] = losses
    np.save("noisy_losses_epsilon_d=6_const_classifier2", noisy)
    np.save("noiseless_losses_epsilon_d=6_const_classifier2", noiseless)
    """
    np.save("noisy_test.npy", noisy)
    a=noisy
    adv_train = np.zeros((runs, eez))
    adv_test = np.zeros((runs, eez))
    train = np.zeros((runs, eez))
    test = np.zeros((runs, eez))
    for i in range(runs):
        for j in range(eez):
            adv_train[i][j] = a[i][j][0]
            train[i][j] = a[i][j][1]
            adv_test[i][j] = a[i][j][2]
            test[i][j] = a[i][j][3]

    plt.plot(epsilons, np.mean(adv_test-adv_train,axis=0), color='red')
    plt.plot(epsilons, np.mean(test-train, axis=0), color='green')
    plt.show()
    """


def main():
    runs = 20
    num_layers=4
    lambda_min = 0.011
    l_mins = [0.0, lambda_min]
    d_H = 8
    epsilons = np.linspace(0.001, lambda_min - 0.001, 8)
    eez = len(epsilons)
    inputs = np.load("x_r=30_d=6_m=20.npy")
    embedding='amplitude'
    model = QuantumClassifier(embedding=embedding, data_dim=6, num_layers=num_layers)

    for l_min in l_mins:
        for i in range(runs):
            x_train = torch.tensor(inputs[i])

        if l_min==0.:
            excess = epsilons * d_H**2 / np.sqrt(20)
        else:
            excess = epsilons * d_H / np.sqrt(20)
        states = torch.stack([torch.pow((1 - l_min * d_H) * model.embedding(x).to(torch.complex64) + l_min * torch.eye(d_H), 2) for x
                      in x_train])



        rad = torch.sum(torch.sqrt(torch.abs(torch.linalg.eigvals(torch.sum(states, 0))))).to(torch.float64)/20

        adv_gen = 2*2.5*(rad+excess)
        gen = 2*2.5*rad*np.ones_like(adv_gen)

        if l_min==0.:
            np.save("noiseless_adv_th", adv_gen)
            np.save("noiseless_th", gen)
        else:
            np.save("noisy_adv_th", adv_gen)
            np.save("noisy_adv_th3", gen+5*excess*d_H)
            np.save("noisy_th", gen)

if __name__ == '__main__':
    main()