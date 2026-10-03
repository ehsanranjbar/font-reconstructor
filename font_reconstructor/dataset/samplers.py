from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class FontBalancedBatchSampler(Sampler):
    """
    Batches of `fonts_per_batch` fonts with `samples_per_font` samples each.

    A contrastive loss needs several samples of the same font in a batch to pull together, and samples of
    other fonts to push away. Random batches over many fonts rarely contain two samples of one font.

    :param indices_by_font: for each font the indices of its samples in the dataset
    :param batches_per_epoch: number of batches of an epoch. By default as many as cover the dataset once.
    """

    def __init__(
        self,
        indices_by_font: Sequence[Sequence[int]],
        fonts_per_batch: int,
        samples_per_font: int,
        batches_per_epoch: Optional[int] = None,
    ):
        self.groups = [np.asarray(indices) for indices in indices_by_font if len(indices) > 0]
        if not self.groups:
            raise ValueError("There are no samples to draw batches from.")
        if samples_per_font < 2:
            raise ValueError("samples_per_font has to be at least 2, a font needs a second sample to match.")

        self.fonts_per_batch = min(fonts_per_batch, len(self.groups))
        self.samples_per_font = samples_per_font
        self.batch_size = self.fonts_per_batch * self.samples_per_font

        if batches_per_epoch is None:
            total_samples = sum(len(group) for group in self.groups)
            batches_per_epoch = max(1, total_samples // self.batch_size)
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        # seeded from torch, so that the global seed of a run also fixes the batches
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        for _ in range(self.batches_per_epoch):
            batch = []
            for group_index in rng.choice(len(self.groups), self.fonts_per_batch, replace=False):
                group = self.groups[group_index]
                picked = rng.choice(group, self.samples_per_font, replace=len(group) < self.samples_per_font)
                batch.extend(int(index) for index in picked)
            yield batch

    def __len__(self):
        return self.batches_per_epoch
