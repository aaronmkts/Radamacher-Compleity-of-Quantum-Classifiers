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
from torch.utils.data import TensorDataset, DataLoader

#Hyperparameters
epochs = 1
batch_size = 25 
learning_rate = 0.01 
num_samples = 1000
dimensions = 2 

# Model
name = 'angle'
layers = 2
num_qubits = dimensions if name in ['angle', 'reuploading'] else np.ceil(np.log2(dimensions))


# Data
x, y = gen_data(num_samples, dimensions)
dataset = TensorDataset(x, y) 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
classifier = get_classifier(name)
classifier = classifier(num_qubits=num_qubits, num_layers=layers)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0


    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Move inputs and labels to device
        inputs, labels = inputs, labels

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = classifier(inputs)
        # Flatten outputs and labels to shape [batch_size]
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print statistics every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], '
                  f'Loss: {running_loss / 10:.4f}')
            running_loss = 0.0


print('Training Finished.')

def loss_func(samples, labels):
    expect=classifier(num_qubits=num_qubits, num_layers=layers)(samples)

    return 1/(1+torch.exp(labels*expect))


