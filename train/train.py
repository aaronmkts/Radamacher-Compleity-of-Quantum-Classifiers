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
from datasets import gen_data
from model import get_classifier
import numpy as np

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
samples, labels = gen_data(num_samples, dimensions)
print(samples.shape, labels.shape) # (1000, 2) (1000,)


# Model
classifier = get_classifier(name)
#output = classifier(num_qubits = num_qubits, num_layers=layers)(samples)

#print(output)

def loss_func(samples, labels):
    expect=classifier(num_qubits=num_qubits, num_layers=layers)(samples)

    return 1/(1+torch.exp(labels*expect))

