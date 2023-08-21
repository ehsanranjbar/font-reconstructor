import json
from collections import OrderedDict
from copy import deepcopy
from itertools import repeat
from pathlib import Path
import numpy as np

import pandas as pd
import torch
import torch.nn.functional as F


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


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


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def move_model_to_cpu(model):
    model = deepcopy(model)
    model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    return model.cpu()


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class TopKCosimAccuracy():
    """
    This class calculates clusters for each label of dataset base on averaging
    the embedding of random samples of each label. Then, it calculates the
    accuracy of the model by comparing the output of the model with the
    clusters. The model is assumed to be accurate if true label is in top k most cosine similar clusters of the output.
    """

    def __init__(self, clusters, labels):
        self.clusters = F.normalize(clusters).transpose(0, 1)
        self.labels = labels

    def __call__(self, output, target, k=[5]):
        # find index of elements in target corresponding to labels
        target = np.searchsorted(self.labels, target)
        acc = np.zeros(shape=(len(k), output.shape[0]))
        with torch.no_grad():
            cosim = F.normalize(output) @ self.clusters

            for i, k in enumerate(k):
                _, topk = torch.topk(cosim, k)
                for j in range(output.shape[0]):
                    if target[j] in topk[j]:
                        acc[i, j] = 1

            return acc.mean(axis=1)
