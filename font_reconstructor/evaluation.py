import torch
from tqdm import tqdm

from font_reconstructor.model.metric import TopKCosimAccuracy
from font_reconstructor.utils import MetricTracker


def topk_metric_names(ks):
    return [f'top{k}_acc' for k in ks]


@torch.no_grad()
def compute_font_centroids(model, loader, device, num_fonts, desc='Clustering Labels'):
    """
    Average the latent vectors of each font over the samples of `loader`.

    :param model: model with an `encode` method
    :param loader: loader of sample dicts with `image` and `font_index`
    :return: (centroids, valid). centroids is a tensor (num_fonts, latent_dim). valid is a bool tensor
             (num_fonts,) that is False for fonts without any usable sample.
    """
    model.eval()
    sums = None
    counts = torch.zeros(num_fonts, device=device)
    for batch in tqdm(loader, desc=desc):
        latent = model.encode(batch['image'].to(device))
        font_index = batch['font_index'].to(device)

        # samples with a broken embedding do not take part in the average
        usable = ~torch.isnan(latent).any(dim=1)
        latent, font_index = latent[usable], font_index[usable]

        if sums is None:
            sums = torch.zeros(num_fonts, latent.shape[1], device=device, dtype=latent.dtype)
        sums.index_add_(0, font_index, latent)
        counts.index_add_(0, font_index, torch.ones_like(font_index, dtype=counts.dtype))

    if sums is None:
        raise ValueError("The clustering loader is empty, font centroids can not be computed.")

    centroids = sums / counts.clamp(min=1).unsqueeze(1)
    return centroids, counts > 0


@torch.no_grad()
def evaluate(model, loader, criterion, metric_ftns, device, topk_acc=None, ks=(), on_batch=None,
             desc='Validation'):
    """
    Evaluate the model on a loader of sample dicts.

    :param model: model with `encode` and `decode` methods
    :param criterion: loss function of (output, target)
    :param metric_ftns: metric functions of (output, target)
    :param topk_acc: optional TopKCosimAccuracy. If given, `top{k}_acc` is reported for each k of `ks`.
    :param on_batch: optional callback of (batch_idx, batch, data, target, latent, output) called for each batch
    :return: dict of the loss and metrics averaged over all samples
    """
    model.eval()
    ks = tuple(ks) if topk_acc is not None else ()
    tracker = MetricTracker('loss', *topk_metric_names(ks), *[m.__name__ for m in metric_ftns])

    for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
        data, target = batch['image'].to(device), batch['target'].to(device)
        n = data.shape[0]

        latent = model.encode(data)
        output = model.decode(latent)

        tracker.update('loss', criterion(output, target).item(), n)
        if ks:
            accuracies = topk_acc(latent, batch['font_index'], ks=ks)
            for name, accuracy in zip(topk_metric_names(ks), accuracies):
                tracker.update(name, accuracy, n)
        for met in metric_ftns:
            tracker.update(met.__name__, met(output, target), n)

        if on_batch is not None:
            on_batch(batch_idx, batch, data, target, latent, output)

    return tracker.result()


def build_topk_accuracy(model, clustering_loader, device, num_fonts):
    """
    estimate the font centroids with the clustering loader and return the top-k accuracy metric built on them
    """
    centroids, valid = compute_font_centroids(model, clustering_loader, device, num_fonts)
    return TopKCosimAccuracy(centroids, valid)
