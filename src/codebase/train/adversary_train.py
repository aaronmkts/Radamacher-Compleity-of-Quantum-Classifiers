import sys
import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from ..adversary import generate_adversarial_dataset


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def loss_func(expvals, labels):
    loss = torch.mean(1 / (1 + torch.exp(labels * expvals)))
    return loss

def train_adversary(classifier: torch.nn.Module, train_dataset, learning_rate, batch_size, attack, epsilon):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    train_loss_epoch = 1.
    num_epochs = 0

    while(train_loss_epoch>0.05):
        classifier.train()  # Set model to training mode
        train_loss_epoch = 0.0
        num_epochs += 1

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            labels = (-1) ** labels.float()

            adv_inputs, adv_labels = generate_adversarial_dataset(
                model=classifier,
                inputs=inputs,
                labels=labels,
                loss_func=loss_func,
                attack=attack,
                epsilon=epsilon
            )

            # Forward pass with combined data
            outputs = classifier(adv_inputs)
            outputs = outputs.view(-1)
  
            # Compute loss
            loss = loss_func(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss_epoch += loss.item()

        # Calculate average training loss for the epoch
        train_loss_epoch /= len(train_loader)
        print('Epoch number: ', num_epochs)

    print('Training Finished.')

    return classifier, train_loss_epoch