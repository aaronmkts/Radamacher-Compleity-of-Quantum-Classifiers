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
        # Apply gradient ascent to maximize the loss
        inputs_adv += epsilon * inputs_adv.grad()/torch.linalg.norm(inputs_adv.grad())

    return inputs_adv.detach()