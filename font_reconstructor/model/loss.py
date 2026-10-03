import torch.nn.functional as F


def mse_loss(output, target):
    return F.mse_loss(output, target)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)
