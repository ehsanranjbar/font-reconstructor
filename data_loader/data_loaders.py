from typing import Tuple

import torch
import torch.nn as nn
from torchvision import transforms

from base import BaseDataLoader
from dataset import RandomTextImageDataset


class RTIDataLoader(BaseDataLoader):
    """
    RandomTextImage data loader using BaseDataLoader
    """

    def __init__(
            self,
            fonts_dir: str,
            annotation_file: str,
            random_seed: int = None,
            total_samples: int = 10000,
            font_size: int = 32,
            text_length: Tuple[int, int] = (3, 8),
            text_image_dims: Tuple[int, int] = (128, 32),
            font_fingerprint_dims: Tuple[int, int] = (32, 32),
            group_by_font=False,
            random_augmentions=True,
            batch_size=128,
            shuffle=True,
            validation_split=0.0,
            num_workers=1,
            pin_memory=False,
    ):
        trsfrm = transforms.Compose([
            transforms.ToTensor(),
            *(
                [
                    transforms.RandomAffine(10, translate=(0.1, 0.1), shear=(-5, 5, -5, 5), scale=(0.8, 1.2)),
                    transforms.RandomPerspective(0.1),
                    transforms.GaussianBlur(3),
                    AddGaussianNoise(0, 0.1),
                ] if random_augmentions else []
            ),
            transforms.Normalize((0.5,), (0.5,))
        ])
        target_trsfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.dataset = RandomTextImageDataset(
            fonts_dir,
            annotation_file,
            random_seed,
            total_samples,
            font_size,
            text_length,
            text_image_dims,
            font_fingerprint_dims,
            transform=trsfrm,
            target_transform=target_trsfrm,
            group_by_font=group_by_font,
            cache_fingerprints=True,
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory=pin_memory)


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, input):
        return (input + torch.randn(input.size()) * self.std + self.mean).clamp(0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
