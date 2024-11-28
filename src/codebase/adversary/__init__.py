from .l_1 import L1
from .l_2 import L2 
from .pgd import PGD
from .fgsm import FGSM
from .l_2 import l_2
import torch

def get_attack(name: str):
    match name:
        case "PGD":
            return PGD
        case "FGSM":
            return FGSM
        case "l_1":
            return L1
        case "l_2":
            return L2
        case _:
            raise ValueError(f"Unknown attack: {name}")
        

def generate_adversarial_dataset(model: torch.nn.Module, inputs, labels, loss_func, attack, epsilon):
    model.eval()  # Set model to evaluation mode
    adversary = get_attack(attack)

    # Iterate over the entire dataset provided by data_loader

    inputs_adv = adversary(model, inputs, labels, loss_func, epsilon)

    for inputs, labels in data_loader:
        if attack == "l_1":
            inputs_adv = adversary(model, inputs, labels, epsilon)
        else:
            inputs_adv = adversary(model, inputs, labels, loss_func, epsilon)
        adv_examples.append(inputs_adv)
        adv_labels.append(labels)

    model.train()

    return inputs_adv, labels