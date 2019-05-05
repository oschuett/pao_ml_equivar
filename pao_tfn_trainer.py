# -*- coding: utf-8 -*-

from pao_tfn_dataset import PAODataset
from pao_tfn_net import PAONet

import torch.nn.functional
import torch.utils.data
from se3cnn.point_utils import difference_matrix

def loss_function(xblock_net, xblock_sample):
    #TODO: wrap this into a torch LossFunction
    # We penalize non-unit vectors later, but we are not going to rely on it here.
    xblock_net_unit = torch.nn.functional.normalize(xblock_net)
    #TODO: This might not be ideal as it implicitly foces the pao basis vectors to be orthogonal.
    projector = torch.matmul(torch.t(xblock_net_unit), xblock_net_unit)
    residual = torch.t(xblock_sample) - torch.matmul(projector, torch.t(xblock_sample))
    return torch.mean(torch.pow(residual, 2))

def ortho(xblock):
    #https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Alternatives\n",
    V = torch.t(xblock)
    VV  = torch.matmul(torch.t(V), V)
    L = torch.cholesky(VV)
    L_inv = torch.inverse(L)
    return torch.matmul(L_inv, torch.t(V))

def train_pao_tfn(pao_files, prim_basis_shells, pao_basis_size, kind_name, max_epochs):
    train_dataset = PAODataset(pao_files, kind_name)
    print("Training NN for kind {} using {} samples.".format(kind_name, len(train_dataset)))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)

    net = PAONet(num_kinds=len(prim_basis_shells),
                 pao_basis_size=pao_basis_size,
                 prim_basis_shells=prim_basis_shells[kind_name],
                 num_hidden=1)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    for epoch in range(max_epochs):
        epoch_missmatch = 0
        epoch_penalty = 0
        epoch_loss = 0
        for batch in train_dataloader:
            kind_onehot, coords, sample_indices = batch
            diff_M = difference_matrix(coords)

            # forward pass
            output_net = net(kind_onehot, diff_M)

            missmatch = torch.tensor(0.0)
            penalty = torch.tensor(0.0)

            #TODO: batchify this to speed things up
            for i, idx in enumerate(sample_indices):  # loop over batch
                # We only care about the xblock of the central atom, which we rolled to the front.
                xblock_net = net.decode_xblock(output_net[i,:,0])
                xblock_sample = train_dataset.sample_xblocks[idx]
                missmatch += loss_function(xblock_net, xblock_sample)

                # penalize non-unit basis vector
                penalty += torch.norm(1 - torch.norm(xblock_net, dim=1))

            loss = missmatch + penalty
            epoch_loss += loss.item()
            epoch_missmatch += missmatch.item()
            epoch_penalty += penalty.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%10 == 0:
            print("Epoch: %i  Missmatch: %f  Penalty: %f Loss: %f"%(epoch, epoch_missmatch, epoch_penalty, epoch_loss))

    return net
#EOF
