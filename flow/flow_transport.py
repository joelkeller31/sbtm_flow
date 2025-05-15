from dataclasses import dataclass
from typing import Callable, Tuple, Union
import jax
import numpy as np
import torch 
# import optax
import drifts 
import flow_networks as networks
from sampling import GaussianSample 


State = np.ndarray
Time = float

""" This Document Contains the main FPE Flow Training Process"""


Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""""
TODO: Think about the best way to initialize the newtork/optimizer without splitting it, that seems stupid
ALSO: Implement a particle system method once that's figured out in networks
"""""


class FBTMSim: 
    """ 
    Base Class for Flow Based Transport Modeling Simulations
    Can be customized for more specific applications
    """
    # initial condition fitting
    n_max_init_opt_steps: int
    init_learning_rate: float
    init_ltol: float
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

    # timestepping
    ltol: float
    gtol: float
    n_opt_steps: int
    learning_rate: float

    # network parameters
    n_hidden: int
    n_neurons: int
    act: Callable[[State], State]
    residual_blocks: bool
    interacting_particle_system: bool

    # general simulation parameters
    key: np.ndarray
    params_list: list
    all_samples: dict

    # output information
    output_folder: str
    output_name: str

    def __init__(self, data_dict: dict) -> None:
        self.__dict__ = data_dict.copy()
    
    def initialize_forcing(self) -> None:
        self.forcing = lambda x, t: self.drift(x, t, *self.force_args)


    #  setup flow network here, no need to split it like the SBTM Paper, but generally should split it like they do
    def initialize_network_and_optimizer(self) -> None:
        if self.interacting_particle_system: 
            " define some logic here for if a self interacting particle system is provided"
            pass
        else: 
            self.flow_network = networks.construct_flow_network(
                    self.d,
                    self.n_hidden,
                    self.n_neurons,
                    self.act,
                    is_gradient=False
            )
        self.opt = torch.optim.RAdam(self.flow_network.parameters(), self.learning_rate)


class SequentialFBTM(FBTMSim):
    n_time_steps: int
    use_SDE: bool
    use_ODE: bool
    save_fac: int
    store_fac: int
    means: dict
    covs: dict
    entropies: list
    mask: np.ndarray

    def setup_loss(self):
        pass 

    def setup_loss_fn_args(self) -> Tuple: 
        " original fxn has device argument"
        pass


    def setup_batched_steppers(self): 
        pass


    def step_samples(
        self,
        step,
        t: float,
        samples: np.ndarray,
        SDE_samples: np.ndarray,
    )  -> Tuple[np.ndarray, np.ndarray]:
            pass 
