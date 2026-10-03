import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Tuple

import numpy as np


def cache_key(*parts) -> str:
    """
    short stable hash of everything the cached array depends on
    """
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


class MemmapArray:
    """
    Read-only view of a `.npy` file that is mapped into memory on first access.

    Only the path is pickled, so data loader workers map the same file and share its pages through the OS
    instead of each holding a copy of the array.
    """

    def __init__(self, path):
        self.path = str(path)
        self._array = None

    def _open(self):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode='r')
        return self._array

    @property
    def shape(self):
        return self._open().shape

    def __len__(self):
        return len(self._open())

    def __getitem__(self, index):
        # copy out of the read-only map so that the result is an ordinary writable array
        return np.array(self._open()[index])

    def __getstate__(self):
        return {'path': self.path, '_array': None}


def cached_array(
    cache_dir,
    name: str,
    key: str,
    shape: Tuple[int, ...],
    dtype,
    fill: Callable[[np.ndarray], None],
) -> MemmapArray:
    """
    Open `<cache_dir>/<name>_<key>.npy`, building it first with `fill(array)` if it does not exist.

    The array is written to a temporary file and moved into place once complete, so an interrupted build
    never leaves a partial cache behind.
    """
    path = Path(cache_dir) / f"{name}_{key}.npy"
    if path.exists():
        print(f"Cache found at {path}, loading from cache")
        cached = MemmapArray(path)
        if tuple(cached.shape) == tuple(shape):
            return cached
        print(f"Cache at {path} has shape {cached.shape}, expected {tuple(shape)}. Rebuilding")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        array = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=dtype, shape=tuple(shape))
        fill(array)
        array.flush()
        del array
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"Saved cache to {path}")
    return MemmapArray(path)
