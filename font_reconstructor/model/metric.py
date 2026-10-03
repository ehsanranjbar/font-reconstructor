import torch
import torch.nn.functional as F


def mean_squared_error(output, target):
    with torch.no_grad():
        return torch.mean((output - target) ** 2)


class TopKCosimAccuracy:
    """
    Top-k font identification accuracy in the latent space.

    A sample counts as correct for k if the centroid of its true font is among the k centroids that are most
    cosine similar to its latent vector.

    :param centroids: tensor (fonts, latent_dim), the mean latent vector of each font
    :param valid: optional bool tensor (fonts,), False for fonts whose centroid is unknown. They never match.
    """

    def __init__(self, centroids, valid=None):
        self.centroids = F.normalize(centroids, dim=1).transpose(0, 1)
        self.valid = valid

    @torch.no_grad()
    def __call__(self, latent, font_index, ks=(5,)):
        """
        :param latent: tensor (batch, latent_dim)
        :param font_index: tensor (batch,), index of the true font of each sample
        :param ks: the k values to evaluate
        :return: list with the accuracy of the batch for each k
        """
        cosim = F.normalize(latent, dim=1) @ self.centroids
        if self.valid is not None:
            cosim = cosim.masked_fill(~self.valid, float('-inf'))

        max_k = min(max(ks), cosim.shape[1])
        topk = cosim.topk(max_k, dim=1).indices
        hits = topk == font_index.to(topk.device).unsqueeze(1)
        if self.valid is not None:
            hits = hits & self.valid[topk]

        return [hits[:, :k].any(dim=1).float().mean().item() for k in ks]
