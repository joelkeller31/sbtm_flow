import numpy as onp
from typing import Optional, Tuple, Callable, Union
import jax
from dataclasses import dataclass
import numpy as np 
import torch 

"""" In this class, we define our network structure """""


"""""
TODO: Figure out the is_gradient argrument, also, figure out the interacting particle system setup
      After, update the FBTM base class 
"""""




class ResidualBlock(torch.nn.Module): 
    def __init__(self, neurons: int, activation: Callable):
        super().__init__()
        self.linear = torch.nn.Linear(neurons, neurons)
        self.activation = activation

    def forward(self, x):
        return x + self.activation(self.linear(x))


def construct_mlp_layers(
    n_hidden: int,
    n_neurons: int,
    act: Callable,
    output_dim: int,
    residual_blocks: bool = True
) -> list:
    
    layers = []

    residual_activation_function = lambda x: x + act(x)
    for layer in range(n_hidden):
        ## construct layer
        if layer == 0 or not residual_blocks:
            layers = layers + [
                    torch.nn.Linear(n_neurons, n_neurons),
                    act()
                ]
        else:
            layers = layers + [
                   ResidualBlock(n_neurons, act())
                ]


    ## construct output layer
    layers = layers + [torch.nn.Linear(n_neurons, output_dim)]
    return layers






def construct_flow_network(
    d: int,
    n_hidden: int,
    n_neurons: int,
    act: Callable[[np.ndarray], np.ndarray],
    residual_blocks: bool = True,
    is_gradient: bool = True
) -> Tuple[Callable, Callable]:
    """Construct a score network for a simpler system
    that does not consist of interacting particles.

    Args:
        d: System dimension.
        n_hidden: Number of hidden layers in the network.
        n_neurons: Number of neurons per layer.
        act: Activation function.
        residual_blocks: Whether or not to use residual blocks.
        is_gradient: Whether or not to compute the score as the gradient of a potential.
    """
    output_dim = 1 if is_gradient else d
    input_layer = torch.nn.Linear(d, n_neurons)
    hidden_layers = construct_mlp_layers(n_hidden, n_neurons, act, output_dim, residual_blocks)
    
    return torch.nn.Sequential(input_layer, *hidden_layers)


    

