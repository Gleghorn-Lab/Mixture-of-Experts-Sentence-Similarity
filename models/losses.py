import torch
import torch.nn as nn
import torch.nn.functional as F


def mnr_loss(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    normalize: bool = True,
    weights: torch.Tensor = None
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
    loss = F.cross_entropy(logits, targets, reduction='none')
    if weights is not None:
        if weights.ndim != 1 or weights.shape[0] != batch1.shape[0]:
            raise ValueError("weights must be a 1D tensor with length equal to the batch size")
        loss = loss * weights
    loss = loss.mean()
    return loss


def mnr_plus_loss(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    normalize: bool = True,
    weights: torch.Tensor = None
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
    targets = (batch1_similarity + batch2_similarity) / 2.0
    loss1 = F.cross_entropy(logits, targets.argmax(dim=1), reduction='none')
    loss2 = F.cross_entropy(logits.T, targets.T.argmax(dim=1), reduction='none')
    loss = loss1 + loss2
    if weights is not None:
        if weights.ndim != 1 or weights.shape[0] != batch1.shape[0]:
            raise ValueError("weights must be a 1D tensor with length equal to the batch size")
        loss = loss * weights
    loss = loss.mean()
    return loss
