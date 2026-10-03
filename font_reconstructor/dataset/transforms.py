import torch
import torch.nn as nn
from torchvision import transforms


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, input):
        return (input + torch.randn(input.size()) * self.std + self.mean).clamp(0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def image_transform(augment=False):
    """
    uint8 text image to a tensor in [-1, 1], optionally distorted by random augmentations
    """
    augmentations = [
        transforms.RandomAffine(10, translate=(0.1, 0.1), shear=(-5, 5, -5, 5), scale=(0.8, 1.2)),
        transforms.RandomPerspective(0.1),
        transforms.GaussianBlur(3),
        AddGaussianNoise(0, 0.1),
    ] if augment else []

    return transforms.Compose([
        transforms.ToTensor(),
        *augmentations,
        transforms.Normalize((0.5,), (0.5,))
    ])


def target_transform():
    """
    uint8 font fingerprint to a tensor in [-1, 1]
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
