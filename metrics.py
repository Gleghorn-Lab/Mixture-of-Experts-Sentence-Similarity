import torch


def calc_f1(ss, labels, cutoff):
    recall = calc_recall(ss, labels, cutoff)
    precision = calc_precision(ss, labels, cutoff)
    f1 = torch.where((precision + recall) != 0, (2 * precision * recall)/(precision + recall), 0)
    return f1


def calc_f1max(ss, labels, limits=torch.tensor([-1, 1]), increment=0.001):
    cutoffs = torch.arange(limits[0], limits[1]-increment, increment)
    f1_scores = torch.tensor([calc_f1(ss, labels, cutoff.item()) for cutoff in cutoffs])
    max_index = torch.argmax(f1_scores)
    return cutoffs[max_index].item(), f1_scores[max_index].item() # return threshold and f1max


def calc_precision(ss, labels, cutoff):
    tp = torch.sum((ss >= cutoff) & (labels == 1))
    fp = torch.sum((ss >= cutoff) & (labels == 0))
    return tp / (tp + fp)


def calc_recall(ss, labels, cutoff):
    tp = torch.sum((ss >= cutoff) & (labels == 1))
    fn = torch.sum((ss < cutoff) & (labels == 1))
    return tp / (tp + fn)


def calc_accuracy(ss, labels, cutoff):
    tp = torch.sum((ss >= cutoff) & (labels == 1))
    tn = torch.sum((ss < cutoff) & (labels == 0))
    return (tp + tn) / len(labels)


def calc_distance(ss, labels):
    return torch.mean(torch.abs(ss - labels))
