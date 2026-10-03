import torch
import torch.nn.functional as F


def l1_loss(output, target):
    """
    mean absolute error. It keeps glyph edges sharper than the squared error, which averages them out.
    """
    return F.l1_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    Supervised contrastive loss (Khosla et al. 2020) on the cosine similarity of the embeddings.

    Every sample is pulled towards the other samples of its label in the batch and pushed away from all
    samples of other labels. This shapes the embedding for retrieval by cosine similarity, also of labels
    that were never trained on.

    :param embeddings: tensor (batch, dim). They are normalized to unit length here.
    :param labels: tensor (batch,) of class ids, for example font indices
    :param temperature: scale of the similarities, lower values weigh hard cases more
    :return: scalar loss, averaged over the samples that have another sample of their label in the batch.
             Zero if no sample has one.
    """
    embeddings = F.normalize(embeddings, dim=1)
    similarity = embeddings @ embeddings.T / temperature

    batch_size = embeddings.shape[0]
    is_self = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    is_positive = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~is_self

    # log probability of picking each other sample, a sample is never compared with itself
    similarity = similarity.masked_fill(is_self, float('-inf'))
    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)

    positives = is_positive.sum(dim=1)
    has_positive = positives > 0
    if not has_positive.any():
        return embeddings.sum() * 0.0

    mean_log_prob = log_prob.masked_fill(~is_positive, 0.0).sum(dim=1) / positives.clamp(min=1)
    return -mean_log_prob[has_positive].mean()


def discriminator_loss(real_scores, fake_scores):
    """
    Least squares GAN loss of the discriminator: real fingerprints should score 1, drawn ones 0.
    """
    real_loss = F.mse_loss(real_scores, torch.ones_like(real_scores))
    fake_loss = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
    return 0.5 * (real_loss + fake_loss)


def adversarial_loss(fake_scores):
    """
    Least squares GAN loss of the decoder: its fingerprints should score 1, that is pass as real.
    """
    return F.mse_loss(fake_scores, torch.ones_like(fake_scores))
