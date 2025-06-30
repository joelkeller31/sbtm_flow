from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
import torch 
# import optax
import drifts as drifts 
from multi_marginal_transport import MultiMarginalFBTM
import argparse
import matplotlib.pyplot as plt 
State = np.ndarray
Time = float


import time

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######## Configuation Parameters #########
## physical, non-forcing parameters
d = 2
D = 0.25
dt = 1e-3
tf = 5
n = 10
n_time_steps = int(tf / dt)
num_trajectories = 50


## configure random seed
repeatable_seed = True
if repeatable_seed:
    rng = np.random.default_rng(42)
else:
    rng = np.random.default_rng()


## harmonic system
experiment = " Harmonic "
N = 15
mask = np.ones(N*d)
amp = lambda t: 2
freq = 1
compute_mut = lambda t: np.array([amp(t)*np.cos(np.pi*freq*t), amp(t)*np.sin(np.pi*freq*t)])
drift = drifts.harmonic_trap
force_args = (compute_mut, N, d)
mu0 = np.tile(compute_mut(0), N)

Noisy = False 

## initial distribution parameters
sig0 = 0.25



### Set up neural network
n_hidden = 4
n_x_neurons=256
n_t_neurons=256
n_epochs = 1000

act = torch.nn.ReLU

learning_rate = 0.01
batch_size =  64 


def construct_simulation():

    sim_params = {
    "sig0": sig0,
    "mu0": mu0,
    "drift": drift,
    "force_args": force_args,
    "amp": amp,
    "freq": freq,
    "dt": dt,
    "D": D,
    "D_sqrt": np.sqrt(D),
    "n": n,
    "N": N,
    "d": d,
    "Noisy": Noisy,
    "learning_rate": learning_rate,
    "n_hidden": n_hidden,
    "n_x_neurons": n_x_neurons,
    "n_t_neurons": n_t_neurons,
    "n_epochs": n_epochs,
    "act": act,
    "rng": rng,
    "batch_size": batch_size, 
    "n_time_steps": n_time_steps,
    "num_trajectories": num_trajectories,
    "experiment": experiment
    }
    
    tot_params = {** sim_params}
    sim = MultiMarginalFBTM(tot_params)

    return sim


if __name__ == '__main__':
    sim = construct_simulation()
    sim.initialize_network_and_optimizer()
    sim.initialize_forcing()
    trajs = sim.generate_training_trajectories()
    sim.train_flow_matching(trajs)

    initial_conditions=trajs[0:1, :, :]
    t_eval = np.linspace(0, 1, 20)  

    generated_traj = sim.generate_learned_trajectory(initial_conditions, t_eval, dt=0.05)
    sim.plot_trajectories_2d(generated_traj, trajs)
