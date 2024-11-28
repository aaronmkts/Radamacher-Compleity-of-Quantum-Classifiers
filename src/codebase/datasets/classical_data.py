import numpy as np
import matplotlib.pyplot as plt
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

import numpy as np
import torch

pi = np.pi

def gen_data(num_samples, dimensions):

    # Initialize data and labels
    x = np.zeros((num_samples, dimensions))
    y = np.random.randint(0, 2, num_samples)  # Random labels {0, 1}

    for i in range(num_samples):
        # Define the mean for each class
        mu = pi*np.concatenate((np.ones(int(dimensions//2)),
                                    (-np.ones(int(dimensions - dimensions//2))) ** y[i]))/4
        
        # Covariance is identity matrix
        cov = 0.1 * np.eye(dimensions)
        
        # Generate data point
        x[i, :] = np.random.multivariate_normal(mu, cov)

    # Convert to torch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    
    return x, y