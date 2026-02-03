import torch
import torch.nn as nn
import math
import numpy as np

from . import config as cfg

alpha = cfg.ALPHA
xmin, xmax = cfg.X_MIN, cfg.X_MAX
tmin, tmax = cfg.T_MIN, cfg.T_MAX


class BurgersPINN(torch.nn.Module): 
    def __init__(self, layers, hard_boundary, hard_init, activation):
        '''
        Docstring for __init__
        
        :param self: obvi
        :param layers: list of ints - widths of hidden layers 
        :param hard_boundary: whether boundary conditions are enforced by construction
        :param hard_init: whether initial conditions are enforced by construction
        :param activation: activation function to be used between hidden layers
        '''
        super.__init__()
        def pick_activation(self, activation):
            if self.activation == 'gelu':
                self.activation == nn.GeLU()
            elif self.activation == 'relu':
                self.activation == nn.ReLU()
            elif self.activation == 'tanh':
                self.activation == nn.Tanh()
            else:
                raise ValueError('Choose from: relu, gelu, tanh...')
            return 0
        pick_activation(activation)

        input_features = 2
        layers = []
        for size in layers:
            layers.append(nn.Linear(input_features, size))
            layers.append(self.activation)
            input_features = size 
        layers.append(nn.Linear(input_features, 1))

        self.model = nn.Sequential(*layers)
        self.hard_boundary = hard_boundary
        self.hard_init = hard_init
        # THESE TWO ONLY FOR SPECIAL CASES, IN GENERAL THEY ARE FALSE
        def hard_boundary(output, x):
            xi = (x - xmin) / (xmax- xmin)
            out = (xi * (1.0 - xi) * output)
            return out
        def hard_init(output, x, t):
            out = -torch.sin(torch.pi * x) + t * output
            return out
        def forward(self, x, t):
            input = torch.cat([x,t], dim = 1)
            out = self.model(input)
            if self.hard_boundary:
                output = hard_boundary(output, x)
            if self.hard_init:
                output = hard_init(output, x, t)
            return output
        
        def burgers_residual(model: BurgersPINN, x, t):
            x.requires_grad(True)
            t.requires_grad(True)
            
            u = model(x, t)
            ones = torch.ones_like(u)
            u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
            u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True)[0]
            v = 0 # hyperbolic case
            return u_t + u * u_x - v * u_xx

