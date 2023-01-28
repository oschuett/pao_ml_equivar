def loss_function(xblock_net, xblock_sample):
    #TODO: This might not be ideal as it implicitly foces the pao basis vectors to be orthogonal-normal.
    projector = torch.matmul(torch.transpose(xblock_net, 1, 2), xblock_net)
    xblock_sample_t = torch.transpose(xblock_sample, 1, 2)
    residual = xblock_sample_t - torch.matmul(projector, xblock_sample_t)
    return torch.mean(torch.pow(residual, 2))
