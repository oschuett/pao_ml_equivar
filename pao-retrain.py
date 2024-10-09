#!/usr/bin/env python3

# author: Ole Schuett

import argparse
import torch

from e3nn import o3
from pathlib import Path
from torch.utils.data import DataLoader

from pao.model import PaoModel
from pao.dataset import PaoDataset
from pao.training import train_model


# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Re-trains an exiting PAO-ML model.")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model", type=Path, required=True)

    # Training data files are passed as positional arguments.
    parser.add_argument("training_data", type=Path, nargs="+")
    args = parser.parse_args()

    # Load existing model and ignore most cmd arguments.
    metadata = {
        "pao_model_version": "",
        "num_neighbors": "",
        "num_distances": "",
        "num_layers": "",
        "cutoff": "",
        "kind_name": "",
        "feature_kind_names": "",
        "prim_basis_name": "",
        "pao_basis_size": "",
    }
    model_script = torch.jit.load(args.model, _extra_files=metadata)
    assert int(metadata["pao_model_version"].decode("utf8")) >= 1
    print(f"Loaded pre-trained model from file: {args.model}")

    # Load the training data.
    kind_name = metadata["kind_name"].decode("utf8")
    num_neighbors = int(metadata["num_neighbors"].decode("utf8"))
    dataset = PaoDataset(
        kind_name=kind_name, num_neighbors=num_neighbors, files=args.training_data
    )
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    print(f"Found {len(dataset)} training samples of kind '{kind_name}'.")

    # Check compatability between model and training data.
    assert dataset.kind.pao_basis_size == int(metadata["pao_basis_size"].decode("utf8"))
    assert dataset.kind.prim_basis_name == metadata["prim_basis_name"].decode("utf8")
    feature_kind_names_csv = ",".join(dataset.feature_kind_names)
    assert feature_kind_names_csv == metadata["feature_kind_names"].decode("utf8")

    # Train the model.
    train_model(model_script, dataloader, args.epochs)

    # Save the model.
    model_script.save(args.model, _extra_files=metadata)
    print(f"Saved model to file: {args.model}")


# ======================================================================================
main()

# EOF
