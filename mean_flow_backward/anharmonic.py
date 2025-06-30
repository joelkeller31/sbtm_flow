from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
import torch 
# import optax
import drifts as drifts 
from mean_transport import MarginalFBTM
import argparse
import matplotlib.pyplot as plt 
State = np.ndarray
Time = float


import time

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dt = 1e-3
tf = 5
n = 10
n_time_steps = int(tf / dt)
store_fac = 25


## configure random seed
repeatable_seed = True

if repeatable_seed:
    rng = np.random.default_rng(42)
else:
    rng = np.random.default_rng()

#work on this
N = 10  # number of particles
d = 2   # dimension
A = 5 # interaction strength
r = 0.2  # particle size
D = 0.01  # diffusion coefficient
R = np.sqrt(5 *N)*r
B = D/R**2  # trap strength
# B = 20

D_sqrt = np.sqrt(D)


# amp = lambda t: 2
# freq = np.pi

amp = lambda t: 3
freq = 1

drift = drifts.anharmonic_gaussian

circular = True

if bool(circular):
    compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), np.sin(freq*t)])
else:
    compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), 0])

## initial distribution parameters
force_args = (A, r, B, N, d, compute_mut)
mu0 = np.tile(compute_mut(0), N)
assert mu0.shape == (N*d,), f"Bad mu0 shape: {mu0.shape}"
sig0 = 0.01
Noisy = False 

experiment = "Anharmonic"

### Set up neural network
n_hidden = 4
n_x_neurons=256
n_t_neurons=32
n_epochs = 750
learning_rate=0.01

act = torch.nn.SiLU

batch_size =  1024


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
    "n_time_steps": n_time_steps,
    "experiment": experiment,
    "batch_size": batch_size, 
    }
    
    tot_params = {** sim_params}
    sim = MarginalFBTM(tot_params)

    return sim





if __name__ == '__main__':
    # start = time.time()
    sim = construct_simulation()
    sim.initialize_network_and_optimizer()
    sim.initialize_forcing()
    trajs = sim.generate_training_trajectories()
    sim.train_flow_matching(trajs)
    initial_conditions=torch.tensor(trajs[-1, :, :-1]) 
    t_space, positions = sim.backward_euler_simulator(initial_conditions, 25)
    sim.plot_trajectories_2d(trajectory=positions, true_traj_pts=trajs)
