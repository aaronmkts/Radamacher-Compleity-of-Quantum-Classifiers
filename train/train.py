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

# General hyperparameters
epochs = 1
learning_rate = 0.02
epsilon = 3.
# Data hyperparameters
num_samples = 2000
train_split = 0.8
batch_size = 8  # Adjust if necessary
# Model hyperparameters
layers = 2

dimensions_list = [2, 4, 8, 16]
embeddings_list = ['angle', 'amplitude']
loss_differences_dict = {'angle': [], 'amplitude': []}

for name in embeddings_list:
    print(f"\nProcessing embedding: {name}")
    loss_differences = []

    for dimensions in dimensions_list:
        print(f"\nProcessing dimension: {dimensions}")

        # List to store loss differences for multiple runs
        loss_differences_runs = []

        for run in range(5):
            print(f"Run {run + 1}/5")

            # Generate data for the current dimension
            x, y = gen_data(num_samples, dimensions)
            dataset = TensorDataset(x, y)

            # Split the dataset
            train_size = int(train_split * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            if name in ['angle', 'reuploading']:
                num_qubits = dimensions
            else:
                num_qubits = int(np.ceil(np.log2(dimensions)))

            num_reup = 3 * num_qubits * layers // dimensions

            classifier_fn = get_classifier(name)
            if name == 'reuploading':
                classifier = classifier_fn(num_qubits=num_qubits, num_layers=layers, num_reup=num_reup)
            else:
                classifier = classifier_fn(num_qubits=num_qubits, num_layers=layers)

            # Training loop
            def loss_func(expvals, labels):
                loss = torch.mean(1 / (1 + torch.exp(labels * expvals)))
                return loss

            optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

            train_loss_list = []
            test_loss_list = []

            for epoch in range(epochs):
                classifier.train()  # Set model to training mode
                running_loss = 0.0
                train_loss_epoch = 0.0

                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    labels = (-1) ** labels.float()

                    optimizer.zero_grad()

                    # Forward pass with combined data
                    outputs = classifier(inputs)
                    outputs = outputs.view(-1)

                    # Compute loss
                    loss = loss_func(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Accumulate loss
                    running_loss += loss.item()
                    train_loss_epoch += loss.item()

                # Calculate average training loss for the epoch
                train_loss_epoch /= len(train_loader)
                train_loss_list.append(train_loss_epoch)

                # Evaluate on the test set
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

                # Calculate average testing loss for the epoch
                test_loss_epoch /= len(test_loader)
                test_loss_list.append(test_loss_epoch)

                print(f'Epoch [{epoch + 1}/{epochs}] completed. Train Loss: {train_loss_epoch:.4f}, Test Loss: {test_loss_epoch:.4f}')

            print('Training Finished.')

            def FGSM(model, inputs, labels, loss_func, epsilon=0.2):
                # Ensure inputs require gradients
                inputs_adv = inputs.clone().detach().requires_grad_(True)
                labels_adv = labels.clone().detach()

                # Zero gradients
                model.zero_grad()

                # Forward pass
                outputs = model(inputs_adv)
                outputs = outputs.view(-1)

                # Compute loss
                loss = loss_func(outputs, labels_adv)

                # Backward pass
                loss.backward()

                # Update adversarial inputs
                with torch.no_grad():
                    # Apply gradient ascent to maximize the loss
                    inputs_adv += epsilon * inputs_adv.grad.sign()

                return inputs_adv.detach()

            def generate_adversarial_dataset(model, data_loader, loss_func, epsilon=0.001, percentage=0.025):
                model.eval()  # Set model to evaluation mode
                adv_examples = []
                adv_labels = []

                total_samples = len(data_loader.dataset)
                limit_samples = int(percentage * total_samples)
                processed_samples = 0

                for inputs, labels in data_loader:
                    if processed_samples >= limit_samples:
                        break  # Stop processing after reaching the limit of samples

                    # Determine how many samples to process in this batch
                    remaining_samples = limit_samples - processed_samples
                    batch_size = inputs.size(0)
                    if remaining_samples < batch_size:
                        # Take only the required number of samples from this batch
                        inputs = inputs[:remaining_samples]
                        labels = labels[:remaining_samples]
                        batch_size = remaining_samples  # Update batch_size for this iteration

                    # Transform labels if necessary
                    labels_adv = (-1) ** labels.float()
                    inputs_adv = FGSM(model, inputs, labels_adv, loss_func, epsilon=epsilon)
                    adv_examples.append(inputs_adv)
                    adv_labels.append(labels)

                    processed_samples += batch_size

                # Concatenate all adversarial examples and labels
                adv_examples = torch.cat(adv_examples, dim=0)
                adv_labels = torch.cat(adv_labels, dim=0)

                return adv_examples, adv_labels

            # Generate adversarial datasets
            adv_train_samples, adv_train_labels = generate_adversarial_dataset(
                model=classifier,
                data_loader=train_loader,
                loss_func=loss_func,
                epsilon=epsilon,
                percentage=0.025
            )

            adv_test_samples, adv_test_labels = generate_adversarial_dataset(
                model=classifier,
                data_loader=test_loader,
                loss_func=loss_func,
                epsilon=epsilon,
                percentage=1
            )

            # Evaluate on adversarial train samples
            adv_train_dataset = TensorDataset(adv_train_samples, adv_train_labels)
            adv_train_loader = DataLoader(adv_train_dataset, batch_size=batch_size, shuffle=False)

            classifier.eval()  # Ensure the model is in evaluation mode
            adv_train_loss = 0.0

            with torch.no_grad():
                for inputs, labels in adv_train_loader:
                    labels = (-1) ** labels.float()
                    outputs = classifier(inputs)
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)

                    # Compute loss
                    loss = loss_func(outputs, labels)
                    adv_train_loss += loss.item() * inputs.size(0)

            adv_train_loss /= len(adv_train_dataset)
            print(f'Adversarial Train Loss: {adv_train_loss:.4f}')

            # Evaluate on adversarial test samples
            adv_test_dataset = TensorDataset(adv_test_samples, adv_test_labels)
            adv_test_loader = DataLoader(adv_test_dataset, batch_size=batch_size, shuffle=False)

            adv_test_loss = 0.0

            with torch.no_grad():
                for inputs, labels in adv_test_loader:
                    labels = (-1) ** labels.float()
                    outputs = classifier(inputs)
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)

                    # Compute loss
                    loss = loss_func(outputs, labels)
                    adv_test_loss += loss.item() * inputs.size(0)

            adv_test_loss /= len(adv_test_dataset)
            print(f'Adversarial Test Loss: {adv_test_loss:.4f}')

            # Compute the modulus of the difference
            loss_difference = abs(adv_train_loss - adv_test_loss)
            print(f'Modulus of the Loss Difference for dimension {dimensions}: {loss_difference:.4f}')

            # Append the loss difference to the list for this run
            loss_differences_runs.append(loss_difference)

        # After 5 runs, pick the maximum loss difference for this dimension
        max_loss_difference = np.mean(loss_differences_runs)
        loss_differences.append(max_loss_difference)
        print(f'Maximum Modulus of the Loss Difference for dimension {dimensions}: {max_loss_difference:.4f}')

    # After processing all dimensions for this embedding, store the loss_differences
    loss_differences_dict[name] = loss_differences

# Plot the modulus of the loss differences against dimensions for both embeddings
plt.figure()
for name in embeddings_list:
    plt.plot(dimensions_list, loss_differences_dict[name], marker='o', label=name)

plt.xlabel('Dimensions (d)')
plt.ylabel('Modulus of the Loss Difference (Max over 5 runs)')
plt.title('Max Modulus of Loss Difference vs Dimensions')
plt.grid(True)
plt.legend()
plt.show()
