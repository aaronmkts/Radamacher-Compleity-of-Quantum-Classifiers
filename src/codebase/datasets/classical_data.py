import numpy as np
import matplotlib.pyplot as plt
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def gen_data(m, d):
    x = np.zeros((m,d))
    y = np.random.randint(0, 2, m)

    for i in range (m):
        mu = np.concatenate((5*np.ones(int(d/2)),5*(-np.ones(int(d/2))) ** y[i]))
        x[i,:]=np.random.normal(mu, 5*np.ones(d))
        
    x = torch.tensor(x)
    y = torch.tensor(y) 
    return x, y
