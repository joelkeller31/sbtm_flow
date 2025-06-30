import torch
import torch.nn as nn
import numpy as np


class SpaceTimeModel3D(nn.Module):
    def __init__(self, dim, n_x_neurons, activation, hidden, n_t_neurons):
        super().__init__()
        self.time_embed_dim = n_t_neurons
        # Space module - now handles [batch, particles, dim]
        self.x_module = nn.Sequential(
            nn.Linear(dim, n_x_neurons),
            nn.LayerNorm(n_x_neurons),  # Changed to LayerNorm
            activation(),
            nn.Linear(n_x_neurons, n_x_neurons),
            nn.LayerNorm(n_x_neurons),
            activation(),
            nn.Linear(n_x_neurons, n_x_neurons),
        )
        
        # Time module - processes time embeddings
        self.t_module = nn.Sequential(
            nn.Linear(n_t_neurons, n_t_neurons),
            nn.LayerNorm(n_t_neurons),
            activation(),
            nn.Linear(n_t_neurons, n_t_neurons),
        )
        
        # Fusion module - combine features from space and time 
        self.fusion_module = nn.Sequential(
            nn.Linear(n_x_neurons + n_t_neurons, hidden),
            nn.LayerNorm(hidden),
            activation(),
            nn.Linear(hidden, dim)
        )

    def time_embedding(self, t):
        """Create sinusoidal time embeddings."""
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        
        half_dim = self.time_embed_dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0)
        
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.time_embed_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings
    
    def forward(self, x, t):


        batch_size, num_particles, dim = x.shape
        
        # Process spatial coordinates
        x_flat = x.reshape(-1, dim)  # [batch*particles, dim]
        x_features = self.x_module(x_flat)
        x_features = x_features.view(batch_size, num_particles, -1)
        
        # Process time
        t_embed = self.time_embedding(t)  # [batch_size, time_embed_dim]
        t_features = self.t_module(t_embed)  # [batch_size, x_latent_dim]
        
        # Expand time features to match particles
        t_features = t_features.unsqueeze(1).expand(-1, num_particles, -1)
        
        # Combine and process
        combined = torch.cat([x_features, t_features], dim=-1)
        combined_flat = combined.reshape(-1, combined.size(-1))
        
        out_flat = self.fusion_module(combined_flat)
        return out_flat.view(batch_size, num_particles, dim)
    

def construct_3d_space_time_model(dim, n_x_neurons, activation, 
                             hidden, n_t_neurons):

    model = SpaceTimeModel3D(
        dim, n_x_neurons, activation, hidden, n_t_neurons
    )
    return model