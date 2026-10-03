from typing import Optional, Tuple

import numpy as np
import torch
from torchvision import transforms

from .capture import CaptureSimulation, clean_text_image


class TextImageTransform:
    """
    Turns the cached rendering of a text into the tensor the model sees.

    Without a capture simulation the text is cropped tightly and centred, the ideal result of preprocessing.
    With one, the rendering first goes through a random print-photograph-cleanup history.

    :param dims: (width, height) of the model input
    :param capture: the capture simulation, None for clean images
    """

    def __init__(self, dims: Tuple[int, int], capture: Optional[CaptureSimulation] = None):
        self.dims = tuple(dims)
        self.capture = capture

    def __call__(self, image: np.ndarray, seed: Optional[int] = None) -> torch.Tensor:
        """
        :param image: uint8 array of white text on black, at the resolution of the cache
        :param seed: seed of the capture simulation. Without one it follows the random state of torch, which
            data loader workers seed differently.
        :return: tensor (1, height, width) in [-1, 1]
        """
        if self.capture is None:
            result = clean_text_image(image, self.dims)
        else:
            if seed is None:
                seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            result = self.capture(image, np.random.default_rng(seed))

        tensor = torch.from_numpy(np.asarray(result, dtype=np.float32) / 127.5 - 1.0)
        return tensor.unsqueeze(0)


def image_transform(dims: Tuple[int, int], augment: bool = False, **capture_options) -> TextImageTransform:
    """
    :param dims: (width, height) of the model input
    :param augment: simulate the capture of the text with a camera and its cleanup
    :param capture_options: arguments of CaptureSimulation
    """
    capture = CaptureSimulation(dims, **capture_options) if augment else None
    return TextImageTransform(dims, capture)


def target_transform():
    """
    uint8 font fingerprint to a tensor in [-1, 1]
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
