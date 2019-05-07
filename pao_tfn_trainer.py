# -*- coding: utf-8 -*-

from pao_tfn_dataset import PAODataset
from pao_tfn_net import PAONet

import torch.nn.functional
import torch.utils.data
from se3cnn.point_utils import difference_matrix

def loss_function(xblock_net, xblock_sample):
    #TODO: wrap this into a torch LossFunction
    # We penalize non-unit vectors later, but we are not going to rely on it here.
    #xblock_net_unit = torch.nn.functional.normalize(xblock_net)
    #projector = torch.matmul(torch.t(xblock_net_unit), xblock_net_unit)
    #TODO: maybe penalize non-unit basis vector
    #penalty += torch.norm(1 - torch.norm(xblock_net, dim=1))
    #TODO: This might not be ideal as it implicitly foces the pao basis vectors to be orthogonal-normal.
    #
    #TODO: clean up, use less transpose
    projector = torch.matmul(torch.transpose(xblock_net, 1, 2), xblock_net)
    residual = torch.transpose(xblock_sample, 1, 2) - torch.matmul(projector, torch.transpose(xblock_sample, 1, 2))
    return torch.mean(torch.pow(residual, 2))

def ortho(xblock):
    #https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Alternatives\n",
    V = torch.t(xblock)
    VV  = torch.matmul(torch.t(V), V)
    L = torch.cholesky(VV)
    L_inv = torch.inverse(L)
    return torch.matmul(L_inv, torch.t(V))

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
        for batch in train_dataloader:
            central_atom, kinds_onehot, coords, xblock_sample = batch
            batch_size = central_atom.shape[0]
            diff_M = difference_matrix(coords)

            # forward pass
            output_net = net(kinds_onehot, diff_M)

            #TODO: clean up
            foo = output_net[range(batch_size), :, central_atom]
            xblock_net = foo[:, net.xblock_decoder]
            loss = loss_function(xblock_net, xblock_sample)

            epoch_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_dataloader)
        if epoch%20 == 0:
            print("Epoch: {}  Loss: {:g}".format(epoch, epoch_loss))

    return net
#EOF
