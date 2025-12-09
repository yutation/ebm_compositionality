import torch
import torch.nn as nn
import numpy as np
from models_pytorch import ResNet128

# TPU support
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp


def langevin_dynamics(models, labels_list, x_init, num_steps=100, step_lr=100.0, task='combination_figure', device=None):
    """
    Perform Langevin dynamics to generate images by composing multiple EBMs
    
    Args:
        models: list of model instances
        labels_list: list of label tensors, one for each model
        x_init: initial noise image [B, C, H, W]
        num_steps: number of Langevin steps
        step_lr: step size for gradient descent
        task: task type ('combination_figure', 'negation_figure', 'or_figure')
        device: device to run on (TPU or CPU/GPU)
    
    Returns:
        x_mod: generated images [B, C, H, W]
    """
    if device is None:
        device = x_init.device
    
    x_mod = x_init.clone()
    x_mod.requires_grad = True
    torch_xla.sync()
    
    for step in range(num_steps):
        # Add noise
        x_mod = x_mod + torch.randn_like(x_mod) * 0.005
        x_mod = x_mod.detach()
        x_mod.requires_grad = True
        
        # Compute energy based on task
        if task == 'or_figure':
            # OR composition: -log(exp(-e1) + exp(-e2))
            e1 = models[0](x_mod, labels_list[0]) + models[1](x_mod, labels_list[1])
            e2 = models[2](x_mod, labels_list[2]) + models[3](x_mod, labels_list[3])
            scale = 100
            
            # Logsumexp for numerical stability
            stacked = torch.stack([-e1, -e2], dim=1)  # [B, 2, 1]
            e_pos = -torch.logsumexp(scale * stacked, dim=1) / scale
            # Note: original code then overrides with e_pos = e2
            e_pos = e2
            
        elif task == 'negation_figure':
            # Negation: combine models with negation
            e_pos = models[0](x_mod, labels_list[0])
            
            for i in range(1, len(models)):
                if i == 1:
                    e_pos = e_pos - 0.001 * models[i](x_mod, labels_list[i])
                elif i == 2:
                    e_pos = e_pos - 0.001 * models[i](x_mod, labels_list[i])
                else:
                    e_pos = e_pos + models[i](x_mod, labels_list[i])
                    
        else:  # combination_figure (default)
            # Simple addition of all energies
            e_pos = 0
            for i, (model, label) in enumerate(zip(models, labels_list)):
                e_pos = e_pos + model(x_mod, label)
        
        # Compute gradient
        e_pos_sum = e_pos.sum()  # Sum over batch for backward
        x_grad = torch.autograd.grad(e_pos_sum, x_mod)[0]
        
        # Update x with gradient descent
        with torch.no_grad():
            x_mod = x_mod - step_lr * x_grad
            x_mod = torch.clamp(x_mod, 0, 1)
        
        # Mark step for XLA optimization
        if device is not None and 'xla' in str(device):
            torch_xla.sync()
    
    return x_mod.detach()


def combination_figure(models, labels_list, n=16, num_steps=100, step_lr=100.0, device=None):
    """Generate combination figure"""
    if device is None:
        device = xm.xla_device()
    
    # Random initial noise
    x_noise = torch.rand(n, 3, 128, 128).to(device)
    
    # Run Langevin dynamics
    output = langevin_dynamics(models, labels_list, x_noise, 
                               num_steps=num_steps, step_lr=step_lr, 
                               task='combination_figure', device=device)
    
    print(f"Output shape: {output.shape}")
    return output


def negation_figure(models, labels_list, n=16, num_steps=100, step_lr=100.0, device=None):
    """Generate negation figure"""
    if device is None:
        device = xm.xla_device()
    
    # Random initial noise
    x_noise = torch.rand(n, 3, 128, 128).to(device)
    
    # Run Langevin dynamics
    output = langevin_dynamics(models, labels_list, x_noise, 
                               num_steps=num_steps, step_lr=step_lr, 
                               task='negation_figure', device=device)
    
    print(f"Output shape: {output.shape}")
    return output


def conceptcombine(models, labels_list, n=5, num_steps=100, step_lr=100.0, device=None):
    """Generate concept combination grid"""
    from itertools import product
    
    if device is None:
        device = xm.xla_device()
    
    factors = 2
    prod_labels = np.array(list(product(*[[0, 1] for i in range(factors)])))
    print(prod_labels)
    prod_labels = np.reshape(np.tile(prod_labels[:, None, :], (1, n, 1)), (-1, 2))
    
    # Create label tensors
    labels_list_batch = []
    for i in range(len(labels_list)):
        label_tensor = torch.from_numpy(np.eye(2)[prod_labels[:, i]]).float().to(device)
        labels_list_batch.append(label_tensor)
    
    # Random initial noise
    x_noise = torch.rand(prod_labels.shape[0], 3, 128, 128).to(device)
    
    # Run Langevin dynamics
    output = langevin_dynamics(models, labels_list_batch, x_noise, 
                               num_steps=num_steps, step_lr=step_lr, 
                               task='conceptcombine', device=device)
    
    print(f"Output shape: {output.shape}")
    return output


if __name__ == "__main__":
    # Get TPU device
    device = xm.xla_device()
    print(f"Using device: {device}")
    
    # Settings
    num_filters = 64
    num_steps = 100
    step_lr = 100.0
    task = 'combination_figure'  # or 'negation_figure', 'or_figure', 'conceptcombine'
    
    # Example: compose 4 models with different attributes
    # Model indices: [old, male, smiling, wavy_hair]
    select_idx = [1, 0, 1, 1]  # binary attributes
    
    # Initialize models with random weights
    num_models = len(select_idx)
    models = [ResNet128(num_channels=3, num_filters=num_filters, train=False, classes=2) 
              for _ in range(num_models)]
    
    # Move models to TPU and set to eval mode
    for i, model in enumerate(models):
        models[i] = model.to(device)
        models[i].eval()
    
    # Create labels (one-hot encoded) on TPU
    n = 16  # number of samples
    labels_list = []
    for idx in select_idx:
        label = torch.zeros(n, 2).to(device)
        label[:, idx] = 1.0  # one-hot
        labels_list.append(label)
    
    print(f"Running {task} with {num_models} models")
    print(f"Selected indices: {select_idx}")
    xp.start_trace("./traces/test_profile")
    # Run the selected task
    if task == 'conceptcombine':
        result = conceptcombine(models, labels_list, n=5, num_steps=num_steps, step_lr=step_lr, device=device)
    elif task == 'combination_figure':
        result = combination_figure(models, labels_list, n=n, num_steps=num_steps, step_lr=step_lr, device=device)
    elif task == 'negation_figure':
        result = negation_figure(models, labels_list, n=n, num_steps=num_steps, step_lr=step_lr, device=device)
    elif task == 'or_figure':
        # OR figure needs 4 models
        if num_models < 4:
            print("OR figure requires 4 models, adding more...")
            while len(models) < 4:
                new_model = ResNet128(num_channels=3, num_filters=num_filters, train=False, classes=2).to(device)
                models.append(new_model)
                labels_list.append(torch.zeros(n, 2).to(device))
                labels_list[-1][:, 1] = 1.0
        result = negation_figure(models, labels_list, n=n, num_steps=num_steps, step_lr=step_lr, device=device)
    
    # Mark step for XLA
    torch_xla.sync()
    print("Done!")
    print(result.shape)
    xp.stop_trace()

