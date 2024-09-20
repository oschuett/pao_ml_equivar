#!/usr/bin/env python3

# author: Ole Schuett

import argparse
from pao_model import PaoModel
from pao_dataset import PaoDataset, pao_loss_function
from e3nn import o3
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pao_file_utils import parse_pao_file


# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Trains an equivariant PAO-ML model.")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--steps", type=int, default=10000)

    # Hyper-parameters - TODO tune default values
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--distances", type=int, default=10)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=6.0)

    # Training parameters.
    parser.add_argument("--batch", type=int, default=64)

    # Training data files are passed as positional arguments.
    parser.add_argument("training_data", type=Path, nargs="+")
    args = parser.parse_args()

    # Load the training data.
    dataset = PaoDataset(
        kind_name=args.kind, num_neighbors=args.neighbors, files=args.training_data
    )
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    print(f"Found {len(dataset)} training samples.")

    # Irreps of primary basis.
    assert dataset.kind.prim_basis_name == "DZVP-MOLOPT-GTH"  # TODO support more
    prim_basis_specs = {
        "O": "2x0e + 2x1o + 1x2e",  # DZVP-MOLOPT-GTH for Oxygen: two s-shells, two p-shells, one d-shell
        "H": "2x0e + 1x1o",  # DZVP-MOLOPT-GTH for Hydrogen: two s-shells, one p-shell
    }

    # Construct the model.
    model = PaoModel(
        prim_basis_irreps=o3.Irreps(prim_basis_specs[args.kind]),
        pao_basis_size=dataset.kind.pao_basis_size,
        num_kinds=dataset.num_kinds,
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
    optim = torch.optim.Adam(model_script.parameters())
    for step in range(args.steps + 1):
        optim.zero_grad()
        for neighbors_relpos, neighbors_features, label in dataloader:
            pred = model_script(neighbors_relpos, neighbors_features)
            loss = pao_loss_function(pred, label)
            loss.backward()
        if step % 1000 == 0:
            print(f"step: {step:5d} | loss: {loss:.8e}")
        optim.step()
    print(f"Training complete, final loss: {loss:.8e}")

    # Save the model.
    output_fn = f"pao_model_{args.kind}.pt"
    metadata = {"foo": "bar"}  # TODO add more metadata
    model_script.save(output_fn, _extra_files=metadata)
    print(f"Saved model to file: {output_fn}")


# ======================================================================================
main()

# EOF
