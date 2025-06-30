### here's what i want 
import numpy as np
from typing import Callable, Tuple, Union




def generate_training_data(
    n_steps: int,
    dt: float,
    N: int,
    d: int,
    D_sqrt: float,
    x0: np.ndarray,  # Initial positions, shape (N, d)
    forcing: Callable[[np.ndarray, float], np.ndarray],
    rng: np.random.Generator, 
    Noisy: bool
) -> np.ndarray:
    
    ts = np.linspace(0, 1, n_steps + 1)
    trajs = np.zeros((n_steps + 1, N, d + 1))  # Extra dimension for time
    
    # Set initial conditions
    trajs[0, :, :d] = x0  # Position
    trajs[0, :, d] = ts[0]  # Time
    
    for i in range(n_steps):
        # Update positions
        new_positions = euler_maruyama_step(
            particle_pos=trajs[i, :, :d],
            t=ts[i],
            D_sqrt=D_sqrt,
            dt=dt,
            rng=rng,
            noisy=Noisy,
            forcing=forcing
        )
        
        # Store positions and current time
        trajs[i+1, :, :d] = new_positions
        trajs[i+1, :, d] = ts[i+1]

    return trajs # (shape n_points, N, d+1)

def euler_maruyama_step(
    particle_pos: np.ndarray,  
    t: float,
    D_sqrt: float,
    dt: float,
    forcing: Callable[[np.ndarray, float], np.ndarray],
    rng: np.random.Generator,
    noisy: bool = False, 
    mask: np.ndarray = None,
) -> np.ndarray:
    # Compute drift
    # Add noise
    if noisy:
        # Set seed for reproducibility
        noise = rng.normal(size=particle_pos.shape)
        diffusion = np.sqrt(2 * dt) * D_sqrt * noise
        print(f'particle pos shape: {particle_pos.shape}')
        return particle_pos + dt * forcing(particle_pos, t).reshape(particle_pos.shape) + diffusion
    else:
        return particle_pos + dt * forcing(particle_pos, t).reshape(particle_pos.shape)
    



def systematic_sample(t, batch_size):
    # Calculate the interval between samples
    interval = t / batch_size
    
    # Random starting point within first interval
    start = np.random.uniform(0, interval)
    
    # Generate evenly spaced indices
    t_idx = np.round(start + np.arange(batch_size) * interval).astype(int)
    
    t_idx = np.clip(t_idx, 0, t-1)
    
    return t_idx


def batch_data(trajs, batch_size): 

    t, n, dnt = trajs.shape
    t_idx = systematic_sample(t, batch_size)
    # x_idx = np.random.choice(range(n), 5, replace=False)
    x_avg = np.mean(trajs[t_idx, :, 0], axis=1)  # Shape: (32,)
    y_avg = np.mean(trajs[t_idx, :, 1], axis=1)  # Shape: (32,)
    avg_coords = np.column_stack((x_avg, y_avg))  # Shape: (32, 2)
    
    return trajs[t_idx, 0, -1], avg_coords, range(n+1)


def batch_data_across_particles (trajs, batch_size):

    t, n, d = trajs.shape
    t_idx = systematic_sample(t, batch_size)
    
    # Stack x and y coordinates properly to get (batch_size, n, 2)
    batch_coords = np.stack([
        trajs[t_idx, :, 0],  # x coordinates
        trajs[t_idx, :, 1]   # y coordinates
    ], axis=-1)
    
    return trajs[t_idx, 0, -1], batch_coords, range(n)

