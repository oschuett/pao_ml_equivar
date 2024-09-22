#!/usr/bin/env python3

# author: Ole Schuett

import torch
import argparse
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
    parser.add_argument("--model", default="trained_pao_model.pt")

    # Test data files are passed as positional arguments.
    parser.add_argument("test_data", type=Path, nargs="+")
    args = parser.parse_args()

    # Load model.
    metadata = {
        "pao_model_version": "",
        "num_neighbors": "",
        "kind_name": "",
        "prim_basis_name": "",
        "pao_basis_size": "",
    }
    model_script = torch.jit.load(args.model, _extra_files=metadata)
    assert int(metadata["pao_model_version"].decode("utf8")) >= 1
    print(f"Loaded model from file: {args.model}")

    # Load the test data.
    kind_name = metadata["kind_name"].decode("utf8")
    num_neighbors = int(metadata["num_neighbors"].decode("utf8"))
    dataset = PaoDataset(
        kind_name=kind_name, num_neighbors=num_neighbors, files=args.test_data
    )
    print(f"Found {len(dataset)} test samples of kind '{kind_name}'.")

    # Check compatability between model and test data.
    assert dataset.kind.pao_basis_size == int(metadata["pao_basis_size"].decode("utf8"))
    assert dataset.kind.prim_basis_name == metadata["prim_basis_name"].decode("utf8")

    # TODO use dataloader for better performance.
    for neighbors_relpos, neighbors_features, label in dataset:
        pred = model_script(neighbors_relpos, neighbors_features)
        loss = loss_function(pred, label)
        print(f"loss: {loss:.8e}")


# ======================================================================================
main()

# EOF
