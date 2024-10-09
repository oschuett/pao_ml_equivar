#!/usr/bin/env python3

# author: Ole Schuett

import torch
import argparse
from e3nn import o3
from pathlib import Path
from torch.utils.data import DataLoader

from pao.model import PaoModel
from pao.dataset import PaoDataset
from pao.training import train_model


# ======================================================================================
def train(model, dataloader, steps) -> None:
    # Train the model.
    optim = torch.optim.Adam(model.parameters())
    for step in range(steps + 1):
        optim.zero_grad()
        for neighbors_relpos, neighbors_features, label in dataloader:
            pred = model(neighbors_relpos, neighbors_features)
            loss = pao_loss_function(pred, label)
            loss.backward()
        if step % 1000 == 0:
            print(f"step: {step:5d} | loss: {loss:.8e}")
        optim.step()
    print(f"Training complete, final loss: {loss:.8e}")


# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an equivariant PAO-ML model.")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model", type=Path, default=None)

    # Hyper-parameters - TODO tune default values
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--distances", type=int, default=10)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=6.0)

    # Training data files are passed as positional arguments.
    parser.add_argument("training_data", type=Path, nargs="+")
    args = parser.parse_args()

    # Load the training data.
    dataset = PaoDataset(
        kind_name=args.kind, num_neighbors=args.neighbors, files=args.training_data
    )
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    print(f"Found {len(dataset)} training samples of kind '{args.kind}'.")

    # Irreps of primary basis.
    assert dataset.kind.prim_basis_name == "DZVP-MOLOPT-GTH"  # TODO support more
    prim_basis_specs = {
        "O": "2x0e + 2x1o + 1x2e",  # two s-shells, two p-shells, one d-shell
        "H": "2x0e + 1x1o",  # two s-shells, one p-shell
    }

    # Construct the model.
    model = PaoModel(
        prim_basis_irreps=o3.Irreps(prim_basis_specs[args.kind]),
        pao_basis_size=dataset.kind.pao_basis_size,
        num_feature_kinds=len(dataset.feature_kind_names),
        num_neighbors=args.neighbors,
        num_distances=args.distances,
        num_layers=args.layers,
        cutoff=args.cutoff,
    )
    num_model_params = sum(p.numel() for p in model.parameters())
    print(f"PAO-ML model will have {num_model_params} parameters.")

    # Compile the model to TorchScript.
    model_script = torch.jit.script(model)

    # Train the model.
    train_model(model_script, dataloader, args.epochs)

    # Save the model.
    metadata = {
        "pao_model_version": "1",
        "num_neighbors": str(args.neighbors),
        "num_distances": str(args.neighbors),
        "num_layers": str(args.layers),
        "cutoff": str(args.cutoff),
        "kind_name": args.kind,
        "feature_kind_names": ",".join(dataset.feature_kind_names),
        "prim_basis_name": dataset.kind.prim_basis_name,
        "pao_basis_size": str(dataset.kind.pao_basis_size),
    }
    fn = (
        args.model
        or f"{dataset.kind.prim_basis_name}-PAO{dataset.kind.pao_basis_size}-{args.kind}.pt"
    )
    model_script.save(fn, _extra_files=metadata)
    print(f"Saved model to file: {fn}")


# ======================================================================================
main()

# EOF
