import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity


# Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
def MNR_loss(batch1: torch.Tensor, batch2: torch.Tensor):
    """
    batch1, batch2 - both torch.Tensor (batch_size, hidden_size)
    This function takes two batches of vectors and returns the Multiple Negatives Ranking (MNR) loss.
    It uses cosine similarity as the similarity function.
    The output of the similarity function can be multiplied by a scale value.
    """
    scores = pairwise_cosine_similarity(batch1, batch2)
    labels = torch.arange(len(scores), dtype=torch.long, device=scores.device)  # Example batch1[i] should match with batch2[i]
    return F.cross_entropy(scores, labels)


def clip_loss(batch1: torch.Tensor, batch2: torch.Tensor, temp: float = 1.0):
    """
    batch1, batch2 - both torch.Tensor (batch_size, hidden_size)
    This function takes two batches of vectors and returns the clip loss.
    It uses dot product as the similarity function.
    The output of the similarity function can be divided by a learned temperature value.
    """
    logits = (batch1 @ batch2.T) / temp
    batch1_similarity = batch1 @ batch1.T
    batch2_similarity = batch2 @ batch2.T
    targets = F.softmax((batch1_similarity + batch2_similarity) / 2 * temp, dim=-1)
    batch1_loss = F.cross_entropy(logits, targets.argmax(dim=1))
    batch2_loss = F.cross_entropy(logits.T, targets.T.argmax(dim=1))
    loss =  (batch1_loss + batch2_loss) / 2.0
    return loss
