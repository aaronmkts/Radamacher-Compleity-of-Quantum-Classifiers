from .l_1 import l_1
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
            return l_1
        case "l_2":
            return l_2
        case _:
            raise ValueError(f"Unknown attack: {name}")
        

def generate_adversarial_dataset(model: torch.nn.Module, inputs, labels, loss_func, attack, epsilon):
    model.eval()  # Set model to evaluation mode
    adversary = get_attack(attack)

    # Iterate over the entire dataset provided by data_loader
    inputs_adv = adversary(model, inputs, labels, loss_func, epsilon)

    model.train()

    return inputs_adv, labels