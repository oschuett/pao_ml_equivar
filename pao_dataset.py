# author: Ole Schuett

import numpy as np
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
from pao_file_utils import parse_pao_file


# ======================================================================================
def pao_loss_function(prediction, label):
    # This assumes the columns of prediction and label are orthonormal.
    p1 = prediction.transpose(-2, -1) @ prediction
    p2 = label.transpose(-2, -1) @ label
    return (p1 - p2).pow(2).sum()


# ======================================================================================
class PaoDataset(Dataset):
    def __init__(self, kind: str, files: List[Path]):
        self.neighbors = []
        self.features = []
        self.labels = []

        for fn in files:
            kinds, atom2kind, coords, xblocks = parse_pao_file(fn)
            # The input of each node is whether it's an oxygen or not.
            features = (
                np.array([(k == "H", k == "O") for k in atom2kind])
                * len(atom2kind) ** 0.5
            )
            for i, k in enumerate(atom2kind):
                if k == kind:
                    # TODO remove center atom
                    neighbors = coords - coords[i]
                    # The loss_functions requires orthonormal labels.
                    label = np.linalg.svd(xblocks[i], full_matrices=False)[2]
                    self.neighbors.append(torch.tensor(neighbors, dtype=torch.float32))
                    self.features.append(torch.tensor(features, dtype=torch.float32))
                    self.labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.neighbors[idx], self.features[idx], self.labels[idx]


# EOF
