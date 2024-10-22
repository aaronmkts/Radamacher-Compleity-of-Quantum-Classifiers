from l_1 import l_1
from pgd import PGD
from FGSM import FGSM

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