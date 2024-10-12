#!/usr/bin/env python3

# author: Ole Schuett

import torch
import argparse
import numpy as np
from e3nn import o3
from pathlib import Path
from torch.utils.data import DataLoader

from pao.model import PaoModel
from pao.dataset import PaoDataset
from pao.training import loss_function


# ======================================================================================
def main() -> None:
    description = "Validates a given equivariant PAO-ML model against test data."
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model", required=True)

    # Test data files are passed as positional arguments.
    parser.add_argument("test_data", type=Path, nargs="+")
    args = parser.parse_args()

    # Load model.
    model = torch.jit.load(args.model)
    assert model.pao_model_version >= 1
    print(f"Loaded model from file: {args.model}")

    # Load the test data.
    dataset = PaoDataset(
        kind_name=model.kind_name,
        num_neighbors=model.num_neighbors,
        files=args.test_data,
    )
    print(f"Found {len(dataset)} test samples of kind '{model.kind_name}'.")

    # Check compatability between model and test data.
    assert dataset.kind.pao_basis_size == model.pao_basis_size
    assert dataset.kind.prim_basis_name == model.prim_basis_name
    assert all(dataset.feature_kind_names == model.feature_kind_names)

    # Compute losses.
    losses = []
    for neighbors_relpos, neighbors_features, label in dataset:
        inputs = {
            "neighbors_relpos": neighbors_relpos,
            "neighbors_features": neighbors_features,
        }
        outputs = model(inputs)
        loss = loss_function(outputs["xblock"], label)
        losses.append(loss.item())

    print("minimum loss: {:.8e}".format(np.amin(losses)))
    print("median  loss: {:.8e}".format(np.median(losses)))
    print("maximum loss: {:.8e}".format(np.amax(losses)))


# ======================================================================================
main()

# EOF
