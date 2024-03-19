import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, confusion_matrix
from scipy.stats import spearmanr

from models.losses import clip_loss


def align_predictions(predictions, label_ids, id2tag):
    preds = predictions.argmax(dim=-1)
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(id2tag[label_ids[i][j].item()])
                preds_list[i].append(id2tag[preds[i][j].item()])
    return preds_list, out_label_list


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


def compute_metrics_multi_label_classification(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    preds = np.array(preds)
    labels = np.array(labels)

    preds = torch.tensor(preds)
    y_true = torch.tensor(labels, dtype=torch.int)

    probs = F.softmax(preds, dim=-1)
    y_pred = (probs > 0.5).int()

    f1, prec, recall, thres = max_metrics(probs, y_true)
    accuracy = accuracy_score(y_pred.flatten(), y_true.flatten())
    hamming = hamming_loss(y_pred.flatten(), y_true.flatten())
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': prec,
        'recall': recall,
        'hamming_loss': hamming,
        'threshold': thres
    }


def compute_metrics_single_label_classification(p: EvalPrediction):

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids
    try:
        preds = torch.tensor(np.array(preds))
        y_true = torch.tensor(np.array(labels), dtype=torch.int).flatten()
    except:
        preds = torch.tensor(np.concatenate(preds, axis=0))
        y_true = torch.tensor(np.concatenate(labels, axis=0), dtype=torch.int).flatten()
    
    if preds.flatten().size() == y_true.size():
        y_pred = preds.flatten()
    else: 
        preds = preds.reshape(y_true.size(0), -1)
        y_pred = preds.argmax(dim=-1).flatten()
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    fmax = f1_score(y_true, y_pred, average='weighted')
    best_precision = precision_score(y_true, y_pred, average='weighted')
    best_recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'f1': fmax,
        'precision': best_precision,
        'recall': best_recall,
        'accuracy': accuracy,
    }


def compute_metrics_regression(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    logits = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    ss_res = np.sum((labels - logits) ** 2) # TODO maybe add same option for CM and spearman plot
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    spearman_rho, pval = spearmanr(logits, labels)
    return {
        'r_squared': r_squared,
        'spearman_rho': spearman_rho,
        'pval': pval
    }


def compute_metrics_mlm(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    logits = np.array(preds)
    labels = np.array(labels)

    preds = np.argmax(logits, axis=-1)
    valid_indices = (labels != -100)
    valid_preds = preds[valid_indices]
    valid_labels = labels[valid_indices]

    accuracy = np.mean(valid_preds == valid_labels)
    return {'mlm_accuracy': accuracy}


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


