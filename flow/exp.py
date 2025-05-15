from dataclasses import dataclass
from typing import Callable, Tuple, Union
import jax
import numpy as np
import torch 
# import optax
import drifts as drifts 
from flow_transport import FBTMSim, SequentialFBTM
import argparse

State = np.ndarray
Time = float

""" This Document Contains the main FPE Flow Training Process"""


Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



######## Configuation Parameters #########
## physical, non-forcing parameters
d = 2
D = 0.25
dt = 1e-3
tf = 5
n = 100
n_time_steps = int(tf / dt)
store_fac = 25


## configure random seed
repeatable_seed = False
if repeatable_seed:
    key = jax.random.PRNGKey(42)
    np.random.seed(42)
else:
    key = jax.random.PRNGKey(np.random.randint(1000))


## harmonic-harmonic system
N = 50
w = 1
mask = np.ones(N*d)
amp = lambda t: 2
freq = np.pi
compute_mut = lambda t: np.array([amp(t)*np.cos(np.pi*freq*t), amp(t)*np.sin(np.pi*freq*t)])
drift = drifts.harmonic_trap
force_args = (compute_mut, N, d)
mu0 = np.tile(compute_mut(0), N)


## initial distribution parameters
sig0 = 0.25


### setup optimizer
init_learning_rate = 5e-3
init_ltol = 1e-6
ltol = np.inf
gtol = 0.1
n_opt_steps = 25
n_max_init_opt_steps = int(1e4)
use_ODE = True
use_SDE = False


### Set up neural network
n_hidden = 1
n_neurons = 100
act = torch.nn.ReLU
residual_blocks = False
interacting_particle_system = False
lam = 0.0


### output data
base_folder   = '/scratch/nb3397/results/sbtm_results'
system_folder = 'harmonic_test'
output_folder = f'{base_folder}/{system_folder}'
output_name   = f'harmonic_ex.npy'
################################################


def construct_simulation(learning_rate: float):

    sim_params = {
    "n_max_init_opt_steps": n_max_init_opt_steps,
    "init_learning_rate": init_learning_rate,
    "init_ltol": init_ltol,
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
    "ltol": ltol,
    "gtol": gtol,
    "n_opt_steps": n_opt_steps,
    "learning_rate": learning_rate,
    "n_hidden": n_hidden,
    "n_neurons": n_neurons,
    "act": act,
    "residual_blocks": residual_blocks,
    "interacting_particle_system": interacting_particle_system,
    "key": key,
    "params_list": [],
    "all_samples": dict(),
    "output_folder": output_folder,
    "output_name": output_name
    }

    seq_params = { 
        "n_time_steps": n_time_steps,
        "use_ODE":use_ODE,
        "use_SDE":use_SDE,
        "save_fac":250,
        "store_fac":store_fac,
        "mask":mask,
        "entropies":[],
        "means":{'SDE': [], 'learned': []},
        "covs":{'SDE': [], 'learned': []}       
    }
    
    tot_params = {** sim_params, ** seq_params}
    sim = SequentialFBTM(tot_params)

    return sim

def get_simulation_parameters():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(
            description='Run an SBTM simulation from the command line.'
        )
    parser.add_argument('--learning_rate', type=float)
    args = parser.parse_args()

    return args.learning_rate


if __name__ == '__main__':
    sim = construct_simulation(get_simulation_parameters())
    sim.initialize_network_and_optimizer()

