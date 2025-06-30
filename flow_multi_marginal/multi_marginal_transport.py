from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
import torch 
from data import generate_training_data, batch_data
from model import construct_3d_space_time_model
from flow import cubic_interpolation
from training import FlowMatchingTrainer
State = np.ndarray
Time = float

import matplotlib.pyplot as plt

""" This Document Contains the main FPE Flow Training Process"""
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class  MultiMarginalFBTM():

    sig0: float
    mu0: np.ndarray

    # system parameters
    drift: Callable[[State, Time], State]
    force_args: Tuple
    amp: Callable[[Time], float]
    freq: float
    dt: float
    D: np.ndarray
    D_sqrt: np.ndarray
    n: int
    d: int
    N: int
    Noisy: bool
    learning_rate: float

    # general simulation parameters
    rng: np.random.Generator
    


    n_time_steps: int
    batch_size: int

    def __init__(self, data_dict: dict) -> None:
        self.__dict__ = data_dict.copy()
    
    def initialize_forcing(self) -> None:
        self.forcing = lambda x, t: self.drift(x, t, *self.force_args)


    def initialize_network_and_optimizer(self) -> None:
        
        self.network = construct_3d_space_time_model(dim=self.d, activation=self.act, hidden=self.n_hidden, n_x_neurons=self.n_x_neurons, n_t_neurons=self.n_t_neurons)
        self.opt = torch.optim.AdamW(
                    self.network.parameters(), 
                    self.learning_rate, 
                    weight_decay=1e-5
                )
        
    def generate_training_trajectories(self):
        x0 = self.mu0.reshape(self.N, self.d) + self.sig0 * self.rng.normal(size=(self.N, self.d))
        self.trajs = generate_training_data(self.n_time_steps, self.dt, self.N, self.d, 
                            self.D_sqrt, x0=x0, forcing=self.forcing, rng=self.rng, Noisy=self.Noisy)
        
        return self.trajs

    def generate_learned_trajectory(self, start, timepoints, dt =0.05):
        b, p, d = start.shape 
        soln_vec = np.zeros(shape=(timepoints.shape[0]+1, p, 2))
        pos = torch.tensor(start[:, :, :-1], dtype=torch.float32)
        soln_vec[0, :, :] = pos
        for i in range(timepoints.shape[0]): 
            vel = self.network(pos, torch.tensor(timepoints[i], dtype=torch.float32))
            pos += dt * vel
            soln_vec[i+1, :, :] = (pos).detach().numpy()
        return soln_vec


    def plot_trajectories_2d(self, trajectory, true_traj_pts):
        
        if type(trajectory) == "torch.tensor": 
            trajectory = trajectory.detach().numpy() 

        tp, par, dim  = trajectory.shape
        plt.figure(figsize=(12, 10))
        
        for p in range(par):
            plt.plot(trajectory[:, p, 0], trajectory[:, p, 1], 
                    label=f'Particle {p+1} (Learned)', alpha=0.7)
            plt.scatter(true_traj_pts[:, p, 0], true_traj_pts[:, p, 1], label=f'Particle {p+1} (Euler Maruyama)', alpha=0.7)
        # # Mark start and end points
        for p in range(par):
            plt.scatter(trajectory[0, p, 0], trajectory[0, p, 1], 
                    marker='o', color='green', s=100, label='Start' if p == 0 else "")
            plt.scatter(trajectory[-1, p, 0], trajectory[-1, p, 1], 
                    marker='x', color='red', s=100, label='End' if p == 0 else "")
        title =  f'Generated Trajectories for {self.experiment} Experiment'
        plt.title(title)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()


    def train_flow_matching(self, trajs):
        
        trainer = FlowMatchingTrainer(model=self.network, opt = self.opt, train_trajs=trajs, batch_size=self.batch_size)
        self.entropies = trainer.train(num_epochs=self.n_epochs)


    