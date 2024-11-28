import sys
import os

# Add the parent directory to the system path to resolve the import issue
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import  DataLoader

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def loss_func(expvals, labels):
    loss = torch.mean(1 / (1 + torch.exp(labels * expvals)))
    return loss

def train(classifier, train_dataset, epochs, learning_rate, batch_size):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    loss_per_epoch = []

    for epoch in range(epochs):
        classifier.train()  # Set model to training mode
        train_loss_epoch = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            labels = (-1) ** labels.float()

            optimizer.zero_grad()

            # Forward pass with combined data
            outputs = classifier(inputs).view(-1)

            # Compute loss
            loss = loss_func(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss_epoch += loss.item()

        # Average loss for the epoch
        train_loss_epoch /= len(train_loader)
        loss_per_epoch.append(train_loss_epoch)

        print(f'Epoch [{epoch + 1}/{epochs}] completed. Train Loss: {train_loss_epoch:.4f}')

    print('Training Finished.')

    return classifier, np.array(loss_per_epoch)
