# -*- coding: utf-8 -*-

import torch
import numpy as np
from functools import partial

#from se3cnn.convolution import SE3PointConvolution
#from se3cnn.blocks.point_norm_block import PointNormBlock
#from se3cnn.point_kernel import gaussian_radial_function

from ole_se3cnn import SE3PointConvolution, PointNormBlock, gaussian_radial_function

class PAONet(torch.nn.Module):
    def __init__(self, num_kinds, pao_basis_size, prim_basis_shells, num_hidden=1, num_radial=4, max_radius=2.5):
        super().__init__()
        self.num_kinds = num_kinds
        self.prim_basis_shells = prim_basis_shells
        self.pao_basis_size = pao_basis_size

        nonlinearity = lambda x: torch.log(0.5 * torch.exp(x) + 0.5)
        sigma = max_radius / num_radial
        radii = torch.linspace(0, max_radius, steps=num_radial, dtype=torch.float64)
        radial_function = partial(gaussian_radial_function, sigma=2*sigma)
        radii_args = {'radii': radii, 'radial_function': radial_function}

        # Convolutions with Norm nonlinearity layers
        self.layers = torch.nn.ModuleList()

        # features
        input_features = [num_kinds, 0, 0]  # L=0 for atom type as one-hot encoding
        hidden_features = [8, 8, 8] # hidden layer with filters L=0,1,2
        output_features = [i * pao_basis_size for i in prim_basis_shells]

        # input layer
        self.layers.append(PointNormBlock(input_features, hidden_features, activation=nonlinearity, **radii_args))

        # hidden layer
        for _ in range(num_hidden):
            self.layers.append(PointNormBlock(hidden_features, hidden_features, activation=nonlinearity, **radii_args))

        # output layer
        Rs_repr = lambda features: [(m, l) for l, m in enumerate(features)]
        self.layers.append(SE3PointConvolution(Rs_repr(hidden_features), Rs_repr(output_features), **radii_args))

        # build xblock decoder, a lookup table to map the network's 1D output into 2D matrices.
        decoder = [[] for _ in range(self.pao_basis_size)]
        x = 0
        for l, m in enumerate(self.prim_basis_shells):
            for i in range(self.pao_basis_size):
                for j in range(m * (2 * l + 1)):
                    decoder[i].append(x)
                    x += 1
        self.xblock_decoder = torch.as_tensor(decoder)


    def forward(self, kinds_onehot, difference_mat, relative_mask=None):
        output = kinds_onehot
        for layer in self.layers:
            output = layer(output, difference_mat, relative_mask)
        return output[:, self.xblock_decoder, :]


#EOF
