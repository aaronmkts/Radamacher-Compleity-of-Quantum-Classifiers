import sys
import os

# Add the parent directory to the system path to resolve the import issue
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from adversary import get_attack
from train import loss_func

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def generate_adversarial_dataset(model, data_loader, loss_func, attack, epsilon):
    model.eval()  # Set model to evaluation mode
    adv_examples = []
    adv_labels = []
    adversary = get_attack(attack)

    # Iterate over the entire dataset provided by data_loader
    for inputs, labels in data_loader:
        inputs_adv = adversary(model, inputs, labels, loss_func, epsilon)
        adv_examples.append(inputs_adv)
        adv_labels.append(labels)

    # Concatenate all adversarial examples and labels
    adv_examples = torch.cat(adv_examples, dim=0)
    adv_labels = torch.cat(adv_labels, dim=0)
    #return model to training mode
    model.training()

    return adv_examples, adv_labels


def adv_train(classifier, train_dataset, learning_rate, batch_size, attack, epsilon):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    #bla

    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    train_loss_epoch = 1.
    num_epochs = 0

    while(train_loss_epoch>0.05):
        classifier.train()  # Set model to training mode
        train_loss_epoch = 0.0
        num_epochs += 1

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            adv_inputs, adv_labels = generate_adversarial_dataset(
                model=classifier,
                data_loader=train_loader,
                loss_func=loss_func,
                attack=attack,
                epsilon=epsilon
            )

            labels = (-1) ** labels.float()

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