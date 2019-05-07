# -*- coding: utf-8 -*-

import torch
import numpy as np
from functools import partial

from se3cnn.convolution import SE3PointConvolution
from se3cnn.blocks.point_norm_block import PointNormBlock
from se3cnn.point_kernel import gaussian_radial_function

#from se3cnn.SO3 import torch_default_dtype
#from se3cnn.utils import torch_default_dtype
#from se3cnn.non_linearities import NormSoftplus

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

        # xblock decoder
        self.xblock_decoder = self.build_xblock_decoder()


    def forward(self, kinds_onehot, difference_mat, relative_mask=None):
        output = kinds_onehot
        for layer in self.layers:
            output = layer(output, difference_mat, relative_mask)
        #return self.decode_xblock(output)
        return output  # TODO: decode needs to support batching


    # prim = ("s1", "s2", "p1x", "p1y", "p1z", "p2x", "p2y", "p2z", "d1xy", "d1yz", "d1zx", "d1xx", "d1zz")
    # xblock = np.array([["%s,%i"%(x,p) for x in prim ] for p in range(4)])
    # #print(xblock)
    # print(decode_xblock(encode_xblock(xblock, [2, 2, 1]), 4, [2, 2, 1]))
    def decode_xblock(self, xvec):
        """Decodes a 1-D array into a [num_pao, num_prim] 2-D block."""
        xblock = []
        i = 0
        for l, m in enumerate(self.prim_basis_shells):
            n = self.pao_basis_size * m * (2 * l + 1)
            xblock.append(xvec[i:i+n].reshape(self.pao_basis_size, m * (2 * l + 1)))
            i += n
        return torch.cat(xblock, dim=1)

    def build_xblock_decoder(self):
        """Builds a lookup table to decodes a 1-D array into a [num_pao, num_prim] 2-D block."""
        decoder = [[] for _ in range(self.pao_basis_size)]
        x = 0
        for l, m in enumerate(self.prim_basis_shells):
            for i in range(self.pao_basis_size):
                for j in range(m * (2 * l + 1)):
                    decoder[i].append(x)
                    x += 1

                #for j in range(m):
                #    # CP2K and se3cnn enumerate the spherical harmonics differently.
                #    # TODO: find more general mapping
                #    #https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
                #    #https://manual.cp2k.org/trunk/CP2K_INPUT/GLOBAL/PRINT.html#SPHERICAL_HARMONICS
                #    if l==0:
                #        lookup[i].append(x) 
                #    elif l==1:
                #        lookup[i].append(x + 1)  # py
                #        lookup[i].append(x + 2)  # pz
                #        lookup[i].append(x + 0)  # px
                #    elif l==2:
                #        #lookup[i].append(?) # dx2
                #        lookup[i].append(0) # TODO: dummy
                #
                #        lookup[i].append(x) # dxy
                #        lookup[i].append(x + 3) # dxz
                #
                #        #lookup[i].append(?) # dy2
                #        lookup[i].append(0) # TODO: dummy
                #
                #        lookup[i].append(x + 1) # dyz
                #        lookup[i].append(x + 2) # dz2
                #        # left over:  and  dx2 - z2
                #    else:
                #        raise NotImplementedError()
                #    lookup[i].append(x)
                #    x += (2 * l + 1)

        return torch.as_tensor(decoder)


#EOF
