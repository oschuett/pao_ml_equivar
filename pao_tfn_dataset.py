# -*- coding: utf-8 -*-

import torch
import numpy as np
from pao_file_utils import parse_pao_file

# alphabet used for one-hot encoding
KIND_ALPHABET = ("H", "O",)

def encode_kind(atom2kind):
    kinds_onehot = np.zeros((len(KIND_ALPHABET), len(atom2kind)))
    for iatom, kind in enumerate(atom2kind):
        idx = KIND_ALPHABET.index(kind)
        kinds_onehot[idx, iatom] = 1.0
    return kinds_onehot

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

        atom2kind_ref = None
        for fn in self.filenames:
            kinds, atom2kind, coords, xblocks = parse_pao_file(fn)

            # check that atom2kind is consistent across dataset
            if atom2kind_ref is None:
                atom2kind_ref = atom2kind
                self.kinds_onehot = encode_kind(atom2kind)
            assert atom2kind_ref == atom2kind

            for iatom, kind in enumerate(atom2kind):
                if kind != self.kind_name:
                    continue
                self.sample_coords.append(coords)
                self.sample_xblocks.append(torch.as_tensor(xblocks[iatom]))
                self.sample_iatoms.append(iatom)

    def __getitem__(self, idx):
        return self.sample_iatoms[idx], self.kinds_onehot, self.sample_coords[idx], self.sample_xblocks[idx]

    def __len__(self):
        return len(self.sample_xblocks)

#EOF
