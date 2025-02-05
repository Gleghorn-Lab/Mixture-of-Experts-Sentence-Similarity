import torch
import torch.nn.functional as F
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score


def calculate_max_metrics(ss, labels, cutoff):

    ss, labels = ss.float(), labels.float()
    tp = torch.sum((ss >= cutoff) & (labels == 1.0))
    fp = torch.sum((ss >= cutoff) & (labels == 0.0))
    fn = torch.sum((ss < cutoff) & (labels == 1.0))
    precision_denominator = tp + fp
    precision = torch.where(precision_denominator != 0, tp / precision_denominator, torch.tensor(0.0))
    recall_denominator = tp + fn
    recall = torch.where(recall_denominator != 0, tp / recall_denominator, torch.tensor(0.0))
    f1 = torch.where((precision + recall) != 0, (2 * precision * recall) / (precision + recall), torch.tensor(0.0))
    return f1, precision, recall


def max_metrics(ss, labels, increment=0.01):
    ss = torch.clamp(ss, -1.0, 1.0)
    min_val = ss.min().item()
    max_val = 1
    if min_val >= max_val:
        min_val = 0
    cutoffs = torch.arange(min_val, max_val, increment)
    metrics = [calculate_max_metrics(ss, labels, cutoff.item()) for cutoff in cutoffs]
    f1s = torch.tensor([metric[0] for metric in metrics])
    precs = torch.tensor([metric[1] for metric in metrics])
    recalls = torch.tensor([metric[2] for metric in metrics])
    valid_f1s = torch.where(torch.isnan(f1s), torch.tensor(-1.0), f1s)  # Replace NaN with -1 to ignore them in argmax
    max_index = torch.argmax(valid_f1s)
    return f1s[max_index].item(), precs[max_index].item(), recalls[max_index].item(), cutoffs[max_index].item()


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
    # Compute max metrics
    f1, prec, recall, thres = max_metrics(cosine_sim, labels_tensor)
    # Compute accuracy based on the threshold found
    predictions = (cosine_sim > thres).float()
    acc = accuracy_score(predictions.flatten().numpy(), labels.flatten())
    # Compute the mean absolute difference between cosine similarities and labels
    dist = torch.mean(torch.abs(cosine_sim - labels_tensor)).item()
    # Return a dictionary of the computed metrics
    return {
        'accuracy': acc,
        'f1_max': f1,
        'precision_max': prec,
        'recall_max': recall,
        'threshold': thres,
        'distance': dist,
    }


def compute_metrics_sentence_similarity_test(p: EvalPrediction):
    preds = p.predictions
    emb_a, emb_b = preds[0], preds[1]
    # Convert embeddings to tensors
    emb_a_tensor = torch.tensor(emb_a)
    emb_b_tensor = torch.tensor(emb_b)

    # Compute cosine similarity between the embeddings
    cosine_sim = F.cosine_similarity(emb_a_tensor, emb_b_tensor)
    # Compute average cosine similarity
    avg_cosine_sim = torch.mean(cosine_sim).item()

    # Compute Euclidean distance between the embeddings
    euclidean_dist = torch.norm(emb_a_tensor - emb_b_tensor, p=2, dim=1)
    # Compute average Euclidean distance
    avg_euclidean_dist = torch.mean(euclidean_dist).item()

    # Return a dictionary of the computed metrics
    return {
        'avg_cosine_similarity': avg_cosine_sim,
        'avg_euclidean_distance': avg_euclidean_dist,
    }


def compute_metrics_benchmark(preds, labels):
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    f1, prec, recall, thres = max_metrics(preds, labels)
    avg_pos_sim = torch.mean(preds[labels == 1.0])
    avg_neg_sim = torch.mean(preds[labels == 0.0])
    ratio = torch.abs(avg_pos_sim / (avg_neg_sim + 1e-8))
    return {
        'f1': f1,
        'precision': prec,
        'recall': recall,
        'threshold': thres,
        'ratio': ratio,
        'avg_pos_sim': avg_pos_sim.item(),
        'avg_neg_sim': avg_neg_sim.item(),

    }
