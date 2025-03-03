import torch
import torch.nn as nn
import torch.nn.functional as F


def mnr_loss(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Multiple negatives ranking loss:
    Maximizes the cosine similarity between element i of batch1 and batch2
    while treating the rest of pairs as negatives.
    """
    if normalize:
        batch1 = F.normalize(batch1, p=2, dim=-1)
        batch2 = F.normalize(batch2, p=2, dim=-1)

    logits = batch1 @ batch2.T
    targets = torch.arange(len(batch1), device=batch1.device)
    loss = F.cross_entropy(logits, targets)
    return loss


def mnr_plus_loss(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    normalize: bool = False,
) -> float:
    """
    Computes a variant of the multiple negatives ranking loss by computing
    intra-batch similarities and using their argmax as targets.
    """
    if normalize:
        batch1 = F.normalize(batch1, p=2, dim=-1)
        batch2 = F.normalize(batch2, p=2, dim=-1)

    logits = batch1 @ batch2.T
    batch1_similarity = batch1 @ batch1.T
    batch2_similarity = batch2 @ batch2.T
    targets = F.softmax((batch1_similarity + batch2_similarity) / 2.0, dim=-1)
    loss1 = F.cross_entropy(logits, targets.argmax(dim=1))
    loss2 = F.cross_entropy(logits.T, targets.T.argmax(dim=1))
    loss = (loss1 + loss2) / 2.0
    return loss
