import sys
import os

# Add the parent directory to the system path to resolve the import issue
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
import numpy as np
from datasets import gen_data
from model import get_classifier
from torch.utils.data import TensorDataset, DataLoader, random_split

#General hyperparameters
epochs = 50

learning_rate = 0.01 

#Data hyperparameters
num_samples = 1000
dimensions = 2
train_split = 0.2
batch_size = int(num_samples * train_split)

x, y = gen_data(num_samples, dimensions)
dataset = TensorDataset(x, y) 

train_size = int(train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Model hyperparameters
name = 'angle'
layers = 2
num_qubits = dimensions if name in ['angle', 'reuploading'] else int(np.ceil(np.log2(dimensions)))
num_reup = 3 * num_qubits * layers // dimensions


classifier_fn = get_classifier(name)
if name == 'reuploading':
    classifier = classifier_fn(num_qubits=num_qubits, num_layers=layers, num_reup=num_reup)
else:
    classifier = classifier_fn(num_qubits=num_qubits, num_layers=layers)

# Training loop

def loss_func(expvals, labels):
    # Ensure that expvals and labels are of the same shape
    loss = torch.mean(1 / (1 + torch.exp(labels * expvals)))
    return loss

optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

#ADVERSARY
def PGD(model, inputs, labels, epsilon=0.1, alpha=0.01, num_iter=10):

    delta = torch.zeros_like(inputs, requires_grad=True)
    for t in range(num_iter):
        feats_adv = inputs + delta
        outputs = [model(f) for f in feats_adv]

        # forward & backward pass through the model, accumulating gradients
        l = loss_func(torch.stack(outputs), labels)
        l.backward()

        # use gradients with respect to inputs and clip the results to lie within the epsilon boundary
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

train_loss_list = []
test_loss_list = []

epsilon = 0.05  # Maximum perturbation for adversarial examples
alpha = 0.01   # Step size for PGD
num_iter = 1  # Number of iterations for PGD

for epoch in range(epochs):
    classifier.train()  # Set model to training mode
    running_loss = 0.0
    train_loss_epoch = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        labels = (-1) ** labels.float()

        optimizer.zero_grad()

        # Generate adversarial examples using PGD
        adv_inputs = PGD(classifier, inputs, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter)

        # Forward pass with combined data
        outputs = classifier(adv_inputs)
        outputs = outputs.view(-1)

        # Compute loss
        loss = loss_func(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        train_loss_epoch += loss.item()

        # Print statistics every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    # **Calculate average training loss for the epoch**
    train_loss_epoch /= len(train_loader)
    train_loss_list.append(train_loss_epoch)

    # **Evaluate on the test set**
    classifier.eval()  
    test_loss_epoch = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = (-1) ** labels.float()
            outputs = classifier(inputs)
            outputs = outputs.view(-1)
            labels = labels.view(-1)

            # Compute loss
            loss = loss_func(outputs, labels)
            test_loss_epoch += loss.item()

    # **Calculate average testing loss for the epoch**
    test_loss_epoch /= len(test_loader)
    test_loss_list.append(test_loss_epoch)

    print(f'Epoch [{epoch + 1}/{epochs}] completed. Train Loss: {train_loss_epoch:.4f}, Test Loss: {test_loss_epoch:.4f}\n')

print('Training Finished.')

# **Plot the training and testing losses**
plt.figure()
plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, epochs + 1), test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss over Epochs')
plt.legend()
plt.show()