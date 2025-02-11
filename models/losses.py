import torch
import torch.nn as nn
import torch.nn.functional as F


def mnr_loss(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    temp: float = 1.0,
    normalize: bool = True,
    weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Multiple negatives ranking loss
    Maximizes the cosine similarity between element i of batch1 and batch2 while treating the rest of pairs as negatives
    """
    if normalize:
        batch1 = F.normalize(batch1, p=2, dim=-1)
        batch2 = F.normalize(batch2, p=2, dim=-1)
    logits = (batch1 @ batch2.T) / temp
    targets = torch.arange(len(batch1), device=batch1.device) # (B,)
    loss = F.cross_entropy(logits, targets, reduction='none')
    if weights is not None:
        loss = (loss * weights).sum() / weights.sum()
    else:
        loss = loss.mean()
    return loss


def mnr_plus_loss(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    temp: float = 1.0,
    normalize: bool = True,
    weights: torch.Tensor = None
) -> float:
    """
    batch1, batch2 - both torch.Tensor (batch_size, hidden_size)
    """
    if normalize:
        batch1 = F.normalize(batch1, p=2, dim=-1)
        batch2 = F.normalize(batch2, p=2, dim=-1)
    batch1_similarity = batch1 @ batch1.T
    batch2_similarity = batch2 @ batch2.T
    logits = (batch1 @ batch2.T) / temp
    targets = F.softmax((batch1_similarity + batch2_similarity) / 2 * temp, dim=-1)
    batch1_loss = F.cross_entropy(logits, targets.argmax(dim=1), reduction='none')
    batch2_loss = F.cross_entropy(logits.T, targets.T.argmax(dim=1), reduction='none')
    loss =  (batch1_loss + batch2_loss) / 2.0
    if weights is not None:
        loss = (loss * weights).sum() / weights.sum()
    else:
        loss = loss.mean()
    return loss


def mnr_plus_plus_loss(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    temp_logit: float = 1.0,
    temp_target: float = 1.0,
    normalize: bool = True,
    weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes a symmetric contrastive loss based on KL divergence between the logits' softmax and a soft target
    distribution derived from intra-batch similarities.
    
    Optionally, you can weight individual batch elements differently using the `weights` parameter.
    The weights are applied to the per-sample KL divergence losses (computed before the batch average).
    
    Args:
        batch1 (torch.Tensor): shape (batch_size, hidden_size)
        batch2 (torch.Tensor): shape (batch_size, hidden_size)
        temp_logit (float): temperature scaling factor for the cross-batch logits.
        temp_target (float): temperature scaling factor for computing the target distribution.
        normalize (bool): whether to L2-normalize the input embeddings.
        weights (torch.Tensor, optional): a tensor of shape (batch_size,) containing weights for each sample.
            These weights are applied to the per-sample losses before averaging.
    
    Returns:
        torch.Tensor: scalar loss value.
    """
    # Optionally normalize embeddings (common practice in contrastive learning)
    if normalize:
        batch1 = F.normalize(batch1, p=2, dim=-1)
        batch2 = F.normalize(batch2, p=2, dim=-1)
    
    # Compute cross-batch similarities (logits)
    logits = (batch1 @ batch2.T) / temp_logit

    # Compute intra-batch similarities to form a target similarity matrix
    batch1_similarity = batch1 @ batch1.T
    batch2_similarity = batch2 @ batch2.T
    target_sim = (batch1_similarity + batch2_similarity) / 2.0

    # Create a soft target distribution from the similarity matrix (with its own temperature)
    targets = F.softmax(target_sim * temp_target, dim=-1)

    # Compute the predicted log probabilities from the cross-batch logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_T = F.log_softmax(logits.T, dim=-1)

    # Compute elementwise KL divergence for both directions without reduction.
    # The output shape is (batch_size, batch_size)
    kl1 = F.kl_div(log_probs, targets, reduction='none')
    kl2 = F.kl_div(log_probs_T, targets.T, reduction='none')

    # Sum over the negatives dimension (axis=1) to obtain per-sample losses (shape: batch_size)
    loss1_per_sample = kl1.sum(dim=1)
    loss2_per_sample = kl2.sum(dim=1)
    loss = (loss1_per_sample + loss2_per_sample) / 2.0
    # If weights are provided, apply them to the per-sample losses.
    if weights is not None:
        # Check that weights is a 1D tensor with the same length as the batch size.
        if weights.ndim != 1 or weights.shape[0] != batch1.shape[0]:
            raise ValueError("weights must be a 1D tensor with length equal to the batch size")
        loss = (loss * weights).sum() / weights.sum()
    else:
        loss = loss.mean()
    return loss
