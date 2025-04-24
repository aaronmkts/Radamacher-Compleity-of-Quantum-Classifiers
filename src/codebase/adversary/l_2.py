import torch

def L2(model, inputs, labels, loss_func, epsilon):
    # Ensure inputs require gradients
    inputs_adv = inputs.clone().detach().requires_grad_(True)
    labels_adv = labels.clone().detach()

    # Zero gradients
    model.zero_grad()

    # Forward pass
    outputs = model(inputs_adv)
    outputs = outputs.view(-1)
    
    # Compute loss
    loss = loss_func(outputs, labels_adv)

    # Backward pass
    loss.backward()

    # Update adversarial inputs

    with torch.no_grad():
        grad = inputs_adv.grad  
        grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True) + 1e-8
        perturbation = epsilon * grad / grad_norm  
        inputs_adv = inputs_adv + perturbation  # Apply perturbation

    return inputs_adv.detach()