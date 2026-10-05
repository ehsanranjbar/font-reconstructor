from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class FontBalancedBatchSampler(Sampler):
    """
    Batches of `fonts_per_batch` fonts with `samples_per_font` samples each.

    A contrastive loss needs several samples of the same font in a batch to pull together, and samples of
    other fonts to push away. Random batches over many fonts rarely contain two samples of one font.

    With `samples_per_epoch` an epoch is a part of a larger dataset instead of a pass over all of it: the
    samples of every font are divided into as many parts as the dataset holds epochs, and each epoch draws
    from its own part. A dataset that renders its samples when they are read can be as large as the whole
    run then, so that no sample is trained on in two epochs. The trainer tells the sampler the epoch with
    `set_epoch`, so that a resumed run goes on with the part it stopped at.

    :param indices_by_font: for each font the indices of its samples in the dataset, as arrays or ranges
    :param batches_per_epoch: number of batches of an epoch. By default as many as cover the dataset once.
    :param samples_per_epoch: number of samples of an epoch, see above. It sets `batches_per_epoch`.
    """

    def __init__(
        self,
        indices_by_font: Sequence[Sequence[int]],
        fonts_per_batch: int,
        samples_per_font: int,
        batches_per_epoch: Optional[int] = None,
        samples_per_epoch: Optional[int] = None,
    ):
        # ranges are kept as they are: those of a dataset with many millions of samples take no memory
        self.groups = [indices if isinstance(indices, range) else np.asarray(indices)
                       for indices in indices_by_font if len(indices) > 0]
        if not self.groups:
            raise ValueError("There are no samples to draw batches from.")
        if samples_per_font < 2:
            raise ValueError("samples_per_font has to be at least 2, a font needs a second sample to match.")

        self.fonts_per_batch = min(fonts_per_batch, len(self.groups))
        self.samples_per_font = samples_per_font
        self.batch_size = self.fonts_per_batch * self.samples_per_font

        total_samples = sum(len(group) for group in self.groups)
        self.parts = 1
        if samples_per_epoch is not None:
            if batches_per_epoch is not None:
                raise ValueError("Give batches_per_epoch or samples_per_epoch, not both.")
            if not 0 < samples_per_epoch <= total_samples:
                raise ValueError(f"samples_per_epoch has to be between 1 and the {total_samples:,} samples there are.")
            self.parts = total_samples // samples_per_epoch
            if min(len(group) for group in self.groups) < self.parts:
                raise ValueError("A font has fewer samples than there are epochs to divide them between.")
            batches_per_epoch = max(1, samples_per_epoch // self.batch_size)
        elif batches_per_epoch is None:
            batches_per_epoch = max(1, total_samples // self.batch_size)
        self.batches_per_epoch = batches_per_epoch
        self.epoch = 1

    def set_epoch(self, epoch: int):
        """
        :param epoch: the epoch that the next iteration is for, counted from 1
        """
        self.epoch = int(epoch)

    def __iter__(self):
        # seeded from torch, so that the global seed of a run also fixes the batches
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        if self.parts > 1:
            yield from self._iter_part(rng, (self.epoch - 1) % self.parts)
            return

        for _ in range(self.batches_per_epoch):
            batch = []
            for group_index in rng.choice(len(self.groups), self.fonts_per_batch, replace=False):
                group = self.groups[group_index]
                picked = rng.choice(len(group), self.samples_per_font, replace=len(group) < self.samples_per_font)
                batch.extend(int(group[position]) for position in picked)
            yield batch

    def _iter_part(self, rng, part):
        """
        the batches of an epoch that draws from part `part` of the samples of every font

        A font goes through its samples of the part in a random order before any of them comes again, so
        the samples of an epoch are all different unless a font is drawn more often than it has samples.
        """
        queues = {}
        for _ in range(self.batches_per_epoch):
            batch = []
            for group_index in rng.choice(len(self.groups), self.fonts_per_batch, replace=False):
                group = self.groups[group_index]
                start, stop = part * len(group) // self.parts, (part + 1) * len(group) // self.parts
                for _ in range(self.samples_per_font):
                    queue = queues.get(group_index)
                    if not queue:
                        queue = queues[group_index] = (start + rng.permutation(stop - start)).tolist()
                    batch.append(int(group[queue.pop()]))
            yield batch

    def __len__(self):
        return self.batches_per_epoch
