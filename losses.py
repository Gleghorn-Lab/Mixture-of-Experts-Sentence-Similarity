import torch
import torch.nn.functional as F


# Adapted from https://github.com/SimiaoZuo/MoEBERT/blob/master/src/transformers/moebert/moe_layer.py
def load_balancing_loss(router_logits: torch.Tensor) -> float:
    # enforces experts should not be used widely more than another
    if router_logits is None:
        return 0
    if isinstance(router_logits, tuple):
        batch_size, num_experts = router_logits[0].shape
        router_logits = torch.cat(router_logits, dim=0) # batch_size * num_hidden_layers, num_experts
    
    router_probs = F.softmax(router_logits, dim=-1)
    gate = torch.argmax(router_probs, dim=-1)
    num_sentences = F.one_hot(gate, num_experts).gt(0).sum(0)
    p = router_probs.mean(0)
    temp = num_sentences.float()
    f = temp / temp.sum(0, keepdim=True) 
    return num_experts * torch.sum(p * f)


def specified_expert_loss(router_logits: torch.Tensor, router_labels: torch.Tensor) -> float:
    # enforces on average the router should route examples to the correct specified expert given the known origin of the input
    if router_logits is None:
        return 0
    if isinstance(router_logits, tuple):
        batch_size, num_experts = router_logits[0].shape
        router_logits = torch.stack(router_logits, dim=2).transpose(1, 2) # batch_size, num_hidden_layers, num_experts
    else:
        print('Must be tuple of all layers router logits')
    
    avg_logits = router_logits.mean(dim=1)
    return F.cross_entropy(avg_logits, router_labels)


# Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
def MNR_loss(batch1: torch.Tensor, batch2: torch.Tensor, scale: float = 1.0) -> float:
    """
    batch1, batch2 - both torch.Tensor (batch_size, hidden_size)
    This function takes two batches of vectors and returns the Multiple Negatives Ranking (MNR) loss.
    It uses cosine similarity as the similarity function.
    The output of the similarity function can be multiplied by a scale value.
    """
    scores = F.cosine_similarity(batch1, batch2, dim=-1) * scale
    labels = torch.arange(len(scores), dtype=torch.float, device=scores.device)  # Example batch1[i] should match with batch2[i]
    return F.cross_entropy(scores, labels)


def clip_loss(batch1: torch.Tensor, batch2: torch.Tensor, temp: float = 1.0) -> float:
    """
    batch1, batch2 - both torch.Tensor (batch_size, hidden_size)
    This function takes two batches of vectors and returns the clip loss.
    It uses cosine similarity as the similarity function.
    The output of the similarity function can be divided by a temperature value.
    """
    logits = (batch1 @ batch2.T) / temp
    batch1_similarity = batch1 @ batch1.T
    batch2_similarity = batch2 @ batch2.T
    targets = F.softmax((batch1_similarity + batch2_similarity) / 2 * temp, dim=-1)
    batch1_loss = F.cross_entropy(logits, targets.argmax(dim=1))
    batch2_loss = F.cross_entropy(logits.T, targets.T.argmax(dim=1))
    loss =  (batch1_loss + batch2_loss) / 2.0
    print(loss)
    return loss
