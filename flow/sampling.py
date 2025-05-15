""" This document holds the sampling classes for the FPE Flow Project """
from dataclasses import dataclass
from typing import Callable, Tuple, Union
import torch 
import numpy as np


def GaussianSample(self, n, N, d): 
    samples_shape = (self. n,self.N*self.d)
    init_samples = self.sig0*np.random.randn(*samples_shape) + self.mu0[None, :]
    return torch.from_numpy(init_samples)


