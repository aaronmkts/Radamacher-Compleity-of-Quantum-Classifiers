import sys
import os


os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..",
    )
)
import torch
import torch.optim as optim


from codebase.datasets import gen_data
from codebase.model import get_classifier
from codebase.adversary import generate_adversarial_dataset


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#control variables
dimensions = 2
embeddings_list = ['angle']
loss_differences_dict = {'angle': []}

#Hyperparams

#General
learning_rate = 0.02

#Data
train_samples = 500
test_samples = 2000
batch_size = 10

#Model
layers = 2

#Adversary
attack='FGSM'
epsilon=0.05


def loss_func(expvals, labels):
    loss = torch.mean(1 / (1 + torch.exp(labels * expvals)))
    return loss


def main(classifier: torch.nn.Module, train_dataset, learning_rate, batch_size, attack, epsilon):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    train_loss_epoch = 1.
    num_epochs = 0


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

        return len(adv_inputs)

if __name__ == '__main__':
    x, y = gen_data(train_samples, dimensions)
    train_dataset = TensorDataset(x, y)

    name = 'angle'
    num_qubits = dimensions

    #instantiate classifier to be trained
    classifier_fn = get_classifier(name)
    classifier = classifier_fn(num_qubits=num_qubits, num_layers=layers)

    siz = main(classifier=classifier,train_dataset=train_dataset, learning_rate=learning_rate
               , batch_size=batch_size, attack=attack, epsilon=epsilon)

    print(siz)



