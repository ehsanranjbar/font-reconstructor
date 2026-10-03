import json
import logging
import random
from collections import OrderedDict
from copy import deepcopy
from itertools import repeat
from pathlib import Path

import numpy as np
import torch

_logger = logging.getLogger(__name__)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def seed_everything(seed):
    """
    fix random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mps_is_available():
    mps = getattr(torch.backends, 'mps', None)
    return mps is not None and mps.is_available()


def prepare_device(n_gpu_use, device='auto'):
    """
    select the device to run on and the gpu indices used for DataParallel.

    :param n_gpu_use: number of GPUs requested. 0 forces CPU.
    :param device: 'auto', 'cuda', 'mps' or 'cpu'. 'auto' prefers CUDA, then Apple's MPS backend, then CPU.
    :return: (torch.device, list of CUDA device ids). The id list is empty unless CUDA is used.
    """
    if device not in ('auto', 'cuda', 'mps', 'cpu'):
        raise ValueError(f"Unknown device '{device}'. Valid options are 'auto', 'cuda', 'mps' and 'cpu'.")

    if n_gpu_use <= 0 or device == 'cpu':
        return torch.device('cpu'), []

    n_cuda = torch.cuda.device_count()
    if device in ('auto', 'cuda') and n_cuda > 0:
        if n_gpu_use > n_cuda:
            _logger.warning("The number of GPU's configured to use is %d, but only %d are available on this machine.",
                            n_gpu_use, n_cuda)
            n_gpu_use = n_cuda
        return torch.device('cuda:0'), list(range(n_gpu_use))

    if device in ('auto', 'mps') and mps_is_available():
        if n_gpu_use > 1:
            _logger.warning("The MPS backend drives a single GPU; n_gpu=%d is treated as 1.", n_gpu_use)
        return torch.device('mps'), []

    _logger.warning("No %s available on this machine, running on CPU.",
                    'GPU' if device == 'auto' else f'{device} device')
    return torch.device('cpu'), []


def unwrap_model(model):
    """
    return the underlying model of a DataParallel wrapper, or the model itself
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def move_model_to_cpu(model):
    return deepcopy(unwrap_model(model)).cpu()


class MetricTracker:
    """
    Running (weighted) averages of a fixed set of named metrics.
    """

    def __init__(self, *keys):
        self._keys = list(keys)
        self.reset()

    def reset(self):
        self._total = {key: 0.0 for key in self._keys}
        self._counts = {key: 0 for key in self._keys}

    def update(self, key, value, n=1):
        self._total[key] += float(value) * n
        self._counts[key] += n

    def avg(self, key):
        return self._total[key] / self._counts[key] if self._counts[key] else 0.0

    def result(self):
        return {key: self.avg(key) for key in self._keys}
