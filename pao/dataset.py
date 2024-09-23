# author: Ole Schuett

import torch
import numpy as np
import scipy.spatial
from pathlib import Path
from typing import List
from torch.utils.data import Dataset

from .io import parse_pao_file


# ======================================================================================
class PaoDataset(Dataset):
    def __init__(self, kind_name: str, num_neighbors: int, files: List[Path]):
        self.neighbors_relpos = []
        self.neighbors_features = []
        self.labels = []

        # Load kinds from the first training data file.
        kinds = parse_pao_file(files[0]).kinds
        kind_names = np.array(sorted(kinds.keys()))
        self.num_kinds = len(kinds)
        self.kind = kinds[kind_name]
        self.kind_name = kind_name

        as_tensor = lambda x: torch.tensor(np.array(x, dtype=np.float32))
        # Load all training data files.
        for fn in files:
            f = parse_pao_file(fn)

            # Build  k-d tree of atom positions.
            assert num_neighbors <= f.coords.shape[0]
            assert np.all(f.cell == np.diag(np.diagonal(f.cell)))
            boxsize = np.diagonal(f.cell)
            kdtree = scipy.spatial.KDTree(np.mod(f.coords, boxsize), boxsize=boxsize)
            # alternative: https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html

            for i, k in enumerate(f.atom2kind):
                if k == kind_name:
                    # Find indicies of neighbor atoms.
                    neighbors = kdtree.query(f.coords[i], num_neighbors)[1]

                    # Compute relative positions of neighbor atoms.
                    neighbors_relpos = [f.coords[j] - f.coords[i] for j in neighbors]

                    # Features of neighbor atoms is the one-hot encoding of their kind.
                    neighbors_kinds = np.array([f.atom2kind[j] for j in neighbors])
                    neighbors_features = [k == kind_names for k in neighbors_kinds]
                    self.neighbors_relpos.append(as_tensor(neighbors_relpos))
                    self.neighbors_features.append(as_tensor(neighbors_features))

                    # Orthonormalize labels as it's required for the loss_functions.
                    label = np.linalg.svd(f.xblocks[i], full_matrices=False)[2]
                    self.labels.append(as_tensor(label))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.neighbors_relpos[i], self.neighbors_features[i], self.labels[i]


# EOF
