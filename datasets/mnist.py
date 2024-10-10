import torchvision.transforms as transforms
from torchvision import datasets
import torch
import numpy as np

def get_mnist_dataloaders(batch_size=256):
    transform_list = transforms.Compose([transforms.ToTensor(), transforms.Resize((16, 16))])
    
    train_data = datasets.MNIST(root='data', train=True, transform=transform_list, download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=transform_list)

    # Binary classification for digits 0 and 1
    idx = torch.as_tensor(train_data.targets) == 1
    idx += torch.as_tensor(train_data.targets) == 0
    train_data = torch.utils.data.dataset.Subset(train_data, np.where(idx == 1)[0])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    idx = torch.as_tensor(test_data.targets) == 1
    idx += torch.as_tensor(test_data.targets) == 0
    test_data = torch.utils.data.dataset.Subset(test_data, np.where(idx == 1)[0])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_loader, test_loader