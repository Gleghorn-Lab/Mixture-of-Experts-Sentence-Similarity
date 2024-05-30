import torch


def data_collator(features):
    batch = {key: torch.stack([f[key] for f in features]) for key in features[0]}
    return batch
