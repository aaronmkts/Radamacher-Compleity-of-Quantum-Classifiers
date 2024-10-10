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
from dataset.mnist_dataset import get_mnist_dataloaders
from model import QML_classifier

# Hyperparameters
batch_size = 256
learning_rate = 0.005
epochs = 100

# Get data loaders
train_loader, test_loader = get_mnist_dataloaders(batch_size)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QML_classifier().to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss and accuracy
epoch_loss = []
epoch_accuracy = []

# Training loop
for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    correct = 0
    total = 0
    print(f"Epoch {epoch + 1}")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float).squeeze()  # Ensure labels match output size
        optimizer.zero_grad()
        output = model(images).squeeze()  # Ensure output is squeezed to match labels
        loss = loss_fn(output, labels)
        total_loss += loss.item()
        predict = torch.round(output)
        correct += (predict == labels).sum().item()
        total += labels.size(0)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    print(f"Time for epoch {epoch + 1}: {end_time - start_time:.2f} seconds")
    epoch_loss.append(total_loss / len(train_loader))
    epoch_accuracy.append(correct / total)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}')

# Save the trained model weights
torch.save(model.state_dict(), "qModel.pth")

# Plotting the training loss and accuracy
fig, ax = plt.subplots()
twin1 = ax.twinx()
p1, = ax.plot([k + 1 for k in range(epochs)], epoch_loss, 'r-', label='Loss')
p2, = twin1.plot([k + 1 for k in range(epochs)], epoch_accuracy, 'b-', label='Accuracy')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
twin1.set_ylabel("Accuracy")
ax.legend(handles=[p1, p2])
plt.show()

# Evaluation on test dataset
model.eval()
total_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float).squeeze()  # Ensure labels match output size
        output = model(images).squeeze()  # Ensure output is squeezed to match labels
        loss = loss_fn(output, labels)
        total_loss += loss.item()
        predict = torch.round(output)
        correct += (predict == labels).sum().item()
        total += labels.size(0)
print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {correct / total:.4f}")