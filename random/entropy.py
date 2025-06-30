import torch 
import numpy as np 
from torch.func import jacfwd, jacrev



def compute_batch_entropy(v_t, pos, t):
    pos.requires_grad_(True)
    v = v_t(pos, t)  # [64, 15, 2]
    
    divergence = torch.zeros(v.shape[:-1], device=pos.device)  # [64, 15]
    
    for i in range(v.shape[-1]):
        # Compute gradient of i-th component
        grad = torch.autograd.grad(
            v[..., i].sum(),  # sum to get correct gradients
            pos,
            create_graph=True,
            retain_graph=True
        )[0]
        # Take diagonal elements (dvi/dxi)
        divergence += grad[..., i]
    # print(f'divergence shape: {divergence.shape}')
    return torch.mean(divergence)