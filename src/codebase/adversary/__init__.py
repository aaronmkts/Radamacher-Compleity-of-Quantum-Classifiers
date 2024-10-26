from .l_1 import l_1
from .pgd import PGD
from .fgsm import FGSM
import torch

def get_attack(name: str):
    match name:
        case "PGD":
            return PGD
        case "FGSM":
            return FGSM
        case "l_1":
            return l_1
        case _:
            raise ValueError(f"Unknown attack: {name}")
        

def generate_adversarial_dataset(model: torch.nn.Module, data_loader, loss_func, attack, epsilon):
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
    model.train()

    return adv_examples, adv_labels