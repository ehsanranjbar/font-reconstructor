import torch
import torch.nn.functional as F


def mean_squared_error(output, target):
    with torch.no_grad():
        return torch.mean((output - target) ** 2)


def style_accuracy(logits, style_index):
    """
    share of the samples whose highest scoring style is the true one
    """
    with torch.no_grad():
        return (logits.argmax(dim=1) == style_index).float().mean().item()


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
        cosim = self.similarities(latent)
        max_k = min(max(ks), cosim.shape[1])
        topk = cosim.topk(max_k, dim=1).indices
        hits = topk == font_index.to(topk.device).unsqueeze(1)
        if self.valid is not None:
            hits = hits & self.valid[topk]

        return [hits[:, :k].any(dim=1).float().mean().item() for k in ks]

    @property
    def num_fonts(self):
        return self.centroids.shape[1]

    @torch.no_grad()
    def similarities(self, latent):
        """
        :return: tensor (batch, fonts), the cosine similarity of each latent vector to each font centroid.
                 Fonts without a centroid get minus infinity.
        """
        cosim = F.normalize(latent, dim=1) @ self.centroids
        if self.valid is not None:
            cosim = cosim.masked_fill(~self.valid, float('-inf'))
        return cosim

    @torch.no_grad()
    def ranks(self, latent, font_index, top=3):
        """
        Where the true font of each sample stands among all fonts, ordered by similarity.

        :param top: number of best matching fonts to return
        :return: (rank, top_fonts, top_similarities). rank is a tensor (batch,) with 1 for a sample whose true
                 font is the best match. Fonts tied with the true font count as better, and a true font without
                 a centroid gets the last rank. top_fonts and top_similarities are tensors (batch, top).
        """
        cosim = self.similarities(latent)
        font_index = font_index.to(cosim.device)
        true_similarity = cosim.gather(1, font_index.unsqueeze(1))
        rank = (cosim > true_similarity).sum(dim=1) + 1
        rank = torch.where(torch.isinf(true_similarity.squeeze(1)), torch.full_like(rank, self.num_fonts), rank)

        top_similarities, top_fonts = cosim.topk(min(top, cosim.shape[1]), dim=1)
        return rank, top_fonts, top_similarities
