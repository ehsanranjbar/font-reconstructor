import torch.nn.functional as F


def mse_loss(output, target):
    return F.mse_loss(output, target)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def dice_loss(output, target):
    smooth = 1.

    iflat = output.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
