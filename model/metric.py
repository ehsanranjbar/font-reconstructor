import torch


def mean_squared_error(output, target):
    with torch.no_grad():
        return torch.mean((output - target) ** 2)
