import sys
import os

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..",
    )
)

from codebase.datasets import gen_data
from codebase.train import train
from codebase.model import get_classifier

#Default Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

#seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

############# HYPERPARAMETERS #############

#control variables
dimensions_list = [2]
embeddings_list = ['angle']
loss_differences_dict = {'angle': [], 'amplitude': []}

#General
learning_rate = 0.02
epochs = 100

#Data
train_samples = 500
test_samples = 2000
batch_size = 50

#Model
layers = 16

def loss_func(expvals, labels, alpha = 15):
    loss = torch.mean(1 / (1 + torch.exp(alpha*labels * expvals)))
    return loss

def main():
    for name in embeddings_list:
        print(f"\nProcessing embedding: {name}")
        loss_differences = []

        runs = 5 if name == 'angle' else 20 #Do more uns for amplitude embedding

        for dimensions in dimensions_list:
            print(f"\nProcessing dimension: {dimensions}")

            # List to store loss differences for multiple runs
            loss_differences_runs = []

            # Generate train data for the current dimension
            x, y = gen_data(train_samples, dimensions)
            train_dataset = TensorDataset(x, y)

            if name == 'angle':
                num_qubits = dimensions
            else:
                num_qubits = int(np.ceil(np.log2(dimensions)))

            #instantiate classifier to be trained
            classifier_fn = get_classifier(name)
            classifier = classifier_fn(num_qubits=num_qubits, num_layers=4)

            #standard trainining
            trained_class, train_loss = train(classifier, train_dataset, epochs, learning_rate, batch_size)

            plt.plot(np.arange(1,epochs+1), train_loss)
            plt.show()

if __name__ == '__main__':
    main()