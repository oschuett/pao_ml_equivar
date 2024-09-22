# author: Ole Schuett

import torch
from torch.utils.data import DataLoader

from .model import PaoModel
from .dataset import PaoDataset


# ======================================================================================
def loss_function(prediction, label):
    # This assumes the columns of prediction and label are orthonormal.
    p1 = prediction.transpose(-2, -1) @ prediction
    p2 = label.transpose(-2, -1) @ label
    return (p1 - p2).pow(2).mean()


# ======================================================================================
def train_model(model, dataloader: DataLoader, steps: int) -> None:
    # Train the model.
    optim = torch.optim.Adam(model.parameters())
    for step in range(steps + 1):
        optim.zero_grad()
        for neighbors_relpos, neighbors_features, label in dataloader:
            pred = model(neighbors_relpos, neighbors_features)
            loss = loss_function(pred, label)
            loss.backward()
        if step % 1000 == 0:
            print(f"step: {step:5d} | loss: {loss:.8e}")
        optim.step()
    print(f"Training complete, final loss: {loss:.8e}")


# EOF
