import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Optional
from flow import cubic_interpolation_across_particles, multi_marginal_flow_batch
from data import batch_data_across_particles
from torch.nn.utils import clip_grad_norm_


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass


    def train(self, num_epochs: int, device: torch.device, **kwargs) -> torch.Tensor:
        # Start
        self.model.to(device)
        opt = self.opt
        self.model.train()


        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(epoch, **kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        # Finish
        self.model.eval()

class FlowMatchingTrainer(Trainer):
    def __init__(self, 
                 opt,
                 model: torch.nn.Module, 
                 train_trajs: list,
                 batch_size: int,
                 sigma_max: float = 0.1,
                ):
        
        super().__init__(model)
        self.train_trajs = train_trajs
        self.opt = opt
        self.batch_size = batch_size
        self.sigma_max = sigma_max
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    def get_train_loss(self, epoch: int, **kwargs) -> torch.Tensor:

        # Get batch data
        t, xs, x_idx = batch_data_across_particles(self.train_trajs, self.batch_size) 
        
        
        # Create interpolation polynomials
        p_x, p_y, p_der_x, p_der_y = cubic_interpolation_across_particles.create_interpolation(t=t, xs=xs) 


        flow = multi_marginal_flow_batch(sigma_max=self.sigma_max, x_idx=x_idx,
                                poly_x=p_x, poly_y=p_y,
                                poly_der_x=p_der_x, poly_der_y=p_der_y)

        u_t = torch.tensor(flow.conditional_flow_batch(t, xs), 
                         device=self.device, dtype=torch.float32) # has shape 64 x 2 
        
        xs_tensor = torch.tensor(xs, device=self.device, dtype=torch.float32)
        t_tensor = torch.tensor(t, device=self.device, dtype=torch.float32)
        
        
        v_t = self.model(xs_tensor, t_tensor)
        

        loss = torch.mean((u_t - v_t)**2)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Current loss: {loss.item():.4f}")
        return loss 
    
    
    def train(self, num_epochs: int, **kwargs):
        clip_max_norm= 1.0

        self.model.to(self.device)
        opt = self.opt
        entropies = [] 
        with tqdm(range(num_epochs), desc="Training") as pbar:
            for epoch in pbar:
                opt.zero_grad()
                loss= self.get_train_loss(epoch, **kwargs)
                loss.backward()


                opt.step()
                if clip_max_norm is not None:
                    total_norm = clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=clip_max_norm,
                        norm_type=2.0  # L2 norm
                    )
                    pbar.set_postfix(loss=loss.item(), grad_norm=total_norm.item())
                else:
                    pbar.set_postfix(loss=loss.item())
                
        self.model.eval()
        return entropies 


