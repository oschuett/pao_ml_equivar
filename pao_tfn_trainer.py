# -*- coding: utf-8 -*-

from pao_tfn_dataset import PAODataset
from pao_tfn_net import PAONet

import torch.nn.functional
import torch.utils.data
#from se3cnn.point_utils import difference_matrix
from ole_se3cnn import difference_matrix
from time import time


def loss_function(xblock_net, xblock_sample):
    #TODO: This might not be ideal as it implicitly foces the pao basis vectors to be orthogonal-normal.
    projector = torch.matmul(torch.transpose(xblock_net, 1, 2), xblock_net)
    xblock_sample_t = torch.transpose(xblock_sample, 1, 2)
    residual = xblock_sample_t - torch.matmul(projector, xblock_sample_t)
    return torch.mean(torch.pow(residual, 2))

def train_pao_tfn(pao_files, prim_basis_shells, pao_basis_size, kind_name, num_hidden, max_epochs):
    train_dataset = PAODataset(pao_files, kind_name)
    print("Training NN for kind {} using {} samples.".format(kind_name, len(train_dataset)))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)

    net = PAONet(num_kinds=len(prim_basis_shells),
                 pao_basis_size=pao_basis_size,
                 prim_basis_shells=prim_basis_shells[kind_name],
                 num_hidden=num_hidden)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    for epoch in range(max_epochs):
        epoch_loss = 0
        start_time = time()
        for batch in train_dataloader:
            central_atom, kinds_onehot, coords, xblock_sample = batch
            batch_size = central_atom.shape[0]
            diff_M = difference_matrix(coords)

            # forward pass
            output_net = net(kinds_onehot, diff_M)
            xblock_net = output_net[range(batch_size), :, :, central_atom]
            loss = loss_function(xblock_net, xblock_sample)
            epoch_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_dataloader)
        epoch_time = time() - start_time
        if epoch%20 == 0:
            print("Epoch: {:5d}  Loss: {:0.4e} Time: {:.3f}s".format(epoch, epoch_loss, epoch_time))

    return net
#EOF
