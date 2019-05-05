# -*- coding: utf-8 -*-

import torch
import numpy as np
from pao_file_utils import parse_pao_file

# Find and parse all .pao files.
# Each file corresponds to a molecular configuration, ie. a frame.
# Since the system contains multiple atoms, each .pao file contains multiple samples.
class PAODataset(object):
    def __init__(self, filenames, kind_name):
        self.filenames = filenames
        self.kind_name = kind_name
        self.sample_iatoms = []
        self.sample_coords = []
        self.sample_xblocks = []
        self.sample_compl_projector = []
        self.kind_names = set()

        #TODO split in training and test set
        for fn in self.filenames:
            kinds, atom2kind, coords, xblocks = parse_pao_file(fn)
            self.kind_names.update(kinds.keys())
            for iatom, kind in enumerate(atom2kind):
                if kind != self.kind_name:
                    continue
                self.sample_coords.append(coords)
                xblock_i = torch.from_numpy(xblocks[iatom])
                self.sample_xblocks.append(xblock_i)
                self.sample_iatoms.append(iatom)

                ## orthonormalize xblock_sample's basis vectors (they deviate slightly)
                ##TODO: add a regularization term for this.
                ##
                ##https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Alternatives
                #V = torch.t(xblock_i)  #TODO: use torch.tensor() instead?
                #VV  = torch.matmul(torch.t(V), V)
                ##print(torch.det(VV))
                #L = torch.cholesky(VV)
                #L_inv = torch.inverse(L)
                #U = torch.matmul(V, torch.t(L_inv))
                #projector = torch.matmul(U, torch.t(U))
                #identity = torch.eye(projector.shape[0])
                #compl_projector = identity - projector
                #self.sample_compl_projector.append(compl_projector)

        # assuming kinds and atom2kind are the same across whole training data
        self.kinds_onehot = self.encode_kind(atom2kind)

    def encode_kind(self, atom2kind):
        kinds_enum = list(sorted(self.kind_names))
        kinds_onehot = np.zeros((len(kinds_enum), len(atom2kind)))
        for iatom, kind in enumerate(atom2kind):
            idx = kinds_enum.index(kind)
            kinds_onehot[idx, iatom] = 1.0
        return kinds_onehot

    def __getitem__(self, idx):
        # roll central atom to the front
        iatom = self.sample_iatoms[idx]
        rolled_kinds = np.roll(self.kinds_onehot, shift=-iatom, axis=1)
        rolled_coords =  np.roll(self.sample_coords[idx], shift=-iatom, axis=0)
        return rolled_kinds, rolled_coords, idx

    def __len__(self):
        return len(self.sample_xblocks)

#EOF
