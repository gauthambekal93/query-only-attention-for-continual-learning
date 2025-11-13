# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 13:00:45 2025

@author: gauthambekal93
"""

import torch
import torch.nn as nn

# Simple network
class SmallNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)  # Remove for linear-only experiment
        return self.linear2(x)

# Initialize model and data
model = SmallNet()
x = torch.randn(10, 10)
y_true = torch.randn(10, 1)
loss_fn = nn.MSELoss()

# Forward pass and loss
y_pred = model(x)
loss = loss_fn(y_pred, y_true)

# Focus only on final layer for Hessian
params = list(model.linear2.parameters())
grads = torch.autograd.grad(loss, params, create_graph=True)
grads_flat = torch.cat([g.reshape(-1) for g in grads])

# Compute Hessian (final layer only)
num_params = grads_flat.numel()
H = torch.zeros((num_params, num_params))
for i in range(num_params):
    second_grads = torch.autograd.grad(grads_flat[i], params, retain_graph=True)
    H[i] = torch.cat([g.reshape(-1) for g in second_grads]).detach()

# Effective rank of Hessian
eigenvalues = torch.linalg.eigvalsh(H)
eigenvalues = torch.clamp(eigenvalues, min=1e-12)  # Prevent log(0)
p = eigenvalues / eigenvalues.sum()
entropy = -torch.sum(p * torch.log(p))
effective_rank = torch.exp(entropy)

print("Effective Rank of Hessian (final layer only):", effective_rank.item())
