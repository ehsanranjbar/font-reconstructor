import pytest
import torch

from font_reconstructor.evaluation import compute_font_centroids
from font_reconstructor.model.metric import TopKCosimAccuracy
from font_reconstructor.utils import MetricTracker, prepare_device


class IdentityEncoder(torch.nn.Module):
    def encode(self, x):
        return x


def test_topk_accuracy():
    centroids = torch.eye(4)
    topk_acc = TopKCosimAccuracy(centroids)

    latent = torch.tensor([
        [1.0, 0.5, 0.2, 0.0],  # ranks fonts 0, 1, 2, 3
        [0.0, 0.2, 0.5, 1.0],  # ranks fonts 3, 2, 1, 0
    ])
    font_index = torch.tensor([1, 0])

    top1, top2, top4, top9 = topk_acc(latent, font_index, ks=(1, 2, 4, 9))
    assert top1 == 0.0
    assert top2 == 0.5
    assert top4 == 1.0
    assert top9 == 1.0


def test_topk_accuracy_ignores_fonts_without_centroid():
    valid = torch.tensor([True, False, True])
    topk_acc = TopKCosimAccuracy(torch.eye(3), valid)

    latent = torch.tensor([[0.0, 1.0, 0.0]])
    assert topk_acc(latent, torch.tensor([1]), ks=(3,)) == [0.0]
    assert topk_acc(latent, torch.tensor([0]), ks=(3,)) == [1.0]


def test_font_centroids():
    loader = [
        {'image': torch.tensor([[1.0, 0.0], [3.0, 0.0]]), 'font_index': torch.tensor([0, 0])},
        {'image': torch.tensor([[0.0, 4.0], [float('nan'), 1.0]]), 'font_index': torch.tensor([2, 2])},
    ]
    centroids, valid = compute_font_centroids(IdentityEncoder(), loader, torch.device('cpu'), num_fonts=3)

    assert valid.tolist() == [True, False, True]
    assert centroids[0].tolist() == [2.0, 0.0]
    assert centroids[2].tolist() == [0.0, 4.0]


def test_metric_tracker_weights_by_count():
    tracker = MetricTracker('loss')
    tracker.update('loss', 1.0, n=3)
    tracker.update('loss', torch.tensor(5.0), n=1)
    assert tracker.avg('loss') == 2.0
    assert tracker.result() == {'loss': 2.0}

    tracker.reset()
    assert tracker.avg('loss') == 0.0


def test_prepare_device():
    assert prepare_device(0) == (torch.device('cpu'), [])
    assert prepare_device(1, 'cpu') == (torch.device('cpu'), [])

    device, device_ids = prepare_device(1)
    if torch.cuda.is_available():
        assert device.type == 'cuda' and device_ids == [0]
    elif torch.backends.mps.is_available():
        assert device.type == 'mps' and device_ids == []
    else:
        assert device.type == 'cpu' and device_ids == []

    with pytest.raises(ValueError):
        prepare_device(1, 'gpu')
