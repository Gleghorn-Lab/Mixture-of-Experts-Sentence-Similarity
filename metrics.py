import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import EvalPrediction
from sklearn.metrics import precision_recall_curve


def max_metrics(ss, labels):
    """
    Compute F1, precision, recall and optimal threshold using the precision-recall curve.

    Parameters:
        ss (torch.Tensor or array-like): The prediction scores.
        labels (torch.Tensor or array-like): The binary ground-truth labels (0 or 1).

    Returns:
        best_f1 (float): The maximum F1 score obtained.
        best_precision (float): Precision at the optimal threshold.
        best_recall (float): Recall at the optimal threshold.
        best_threshold (float): The threshold that yields the best F1 score.
    """
    # If using torch tensors, convert them to numpy arrays
    if hasattr(ss, 'detach'):
        ss = ss.detach().cpu().numpy()
    else:
        ss = np.array(ss)
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.array(labels)
    
    # Optionally, you might clamp your scores (if needed)
    ss = np.clip(ss, -1.0, 1.0)
    
    # Compute the precision, recall, and thresholds.
    # Note: thresholds will have shape (len(precision)-1,)
    precision, recall, thresholds = precision_recall_curve(labels, ss)
    
    # Compute F1 score at each threshold candidate.
    # We ignore the last value since thresholds is one element shorter.
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
    optimal_idx = np.argmax(f1_scores)
    best_threshold = thresholds[optimal_idx]
    best_f1 = f1_scores[optimal_idx]
    best_precision = precision[optimal_idx]
    best_recall = recall[optimal_idx]
    
    return best_f1, best_precision, best_recall, best_threshold


def compute_metrics_sentence_similarity(p: EvalPrediction):
    preds = p.predictions
    labels = p.label_ids[-1]
    emb_a, emb_b = preds[0], preds[1]
    # Convert embeddings to tensors
    emb_a_tensor = torch.tensor(emb_a)
    emb_b_tensor = torch.tensor(emb_b)
    labels_tensor = torch.tensor(labels)

    # Compute cosine similarity between the embeddings
    cosine_sim = F.cosine_similarity(emb_a_tensor, emb_b_tensor)
    f1, prec, recall, thres = max_metrics(cosine_sim, labels_tensor)
    
    pairwise_cosine_sim = pairwise_cosine_similarity(emb_a_tensor, emb_b_tensor)
    diag = pairwise_cosine_sim.diagonal().diag_embed()
    sim_pos = pairwise_cosine_sim[diag != 0].mean().item()
    sim_neg = pairwise_cosine_sim[diag == 0].mean().item()
    sim_ratio = abs(sim_pos / (sim_neg + 1e-8))

    return {
        'f1': round(f1, 4),
        'precision': round(prec, 4),
        'recall': round(recall, 4),
        'threshold': round(thres, 4),
        'sim_ratio': round(sim_ratio, 4),
        'pos_sim': round(sim_pos, 4),
        'neg_sim': round(sim_neg, 4),
    }


def compute_metrics_benchmark(preds, labels):
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    f1, prec, recall, thres = max_metrics(preds, labels)
    avg_pos_sim = torch.mean(preds[labels == 1.0])
    avg_neg_sim = torch.mean(preds[labels == 0.0])
    ratio = torch.abs(avg_pos_sim / (avg_neg_sim + 1e-8)).item()
    return {
        'f1': round(f1, 4),
        'precision': round(prec, 4),
        'recall': round(recall, 4),
        'threshold': round(thres, 4),
        'sim_ratio': round(ratio, 4),
        'pos_sim': round(avg_pos_sim.item(), 4),
        'neg_sim': round(avg_neg_sim.item(), 4),
    }
