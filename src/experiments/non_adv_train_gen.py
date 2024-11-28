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
dimensions_list = [2, 4, 8, 16]
embeddings_list = ['angle', 'amplitude']
loss_differences_dict = {'angle': [], 'amplitude': []}

#General
learning_rate = 0.02
epochs = 5

#Data
train_samples = 500
test_samples = 2000
batch_size = 10

#Model
layers = 2

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

            for run in range(runs):
                print(f"Run {run + 1}/{runs}")

                # Generate train data for the current dimension
                x, y = gen_data(train_samples, dimensions)
                train_dataset = TensorDataset(x, y)

                if name == 'angle':
                    num_qubits = dimensions
                else:
                    num_qubits = int(np.ceil(np.log2(dimensions)))

                #instantiate classifier to be trained
                classifier_fn = get_classifier(name)
                classifier = classifier_fn(num_qubits=num_qubits, num_layers=layers)

                #standard trainining
                trained_class, train_loss = train(classifier, train_dataset, epochs, learning_rate, batch_size)

                #generate test data
                x, y = gen_data(test_samples, dimensions)
                test_dataset = TensorDataset(x, y)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                #Evaluate test loss
                test_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        labels = (-1) ** labels.float()
                        outputs = trained_class(inputs)
                        outputs = outputs.view(-1)
                        labels = labels.view(-1)

                        # Compute loss
                        loss = loss_func(outputs, labels)
                        test_loss += loss.item() * inputs.size(0)

                test_loss /= len(test_dataset)

                #mod of test - train loss
                loss_differences_runs.append(test_loss-train_loss)

            #max of loss differences over runs indicates worst case generalization
            loss_differences.append(np.max(loss_differences_runs))

        # After processing all dimensions for this embedding, store the loss_differences
        loss_differences_dict[name] = loss_differences


if __name__ == '__main__':
    main()
    # Plot the modulus of the loss differences against dimensions for both embeddings
    plt.figure()
    for name in embeddings_list:
        plt.plot(dimensions_list, loss_differences_dict[name], marker='o', label=name)

    plt.xlabel('Dimensions (d)')
    plt.ylabel('Test Loss (Max over 5 runs)')
    plt.title('Max Modulus of Loss Difference vs Dimensions')
    plt.grid(True)
    plt.legend()
    plt.show()