import yaml
import torch


def get_yaml(yaml_file):
    if yaml_file == None:
        return None
    else:
        with open(yaml_file, 'r') as file:
            args = yaml.safe_load(file)
        return args


def log_metrics(log_path, metrics, details=None, header=None):
    def log_nested_dict(d, parent_key=''):
        filtered_results = {}
        for k, v in d.items():
            new_key = f'{parent_key}_{k}' if parent_key else k
            if isinstance(v, dict):
                filtered_results.update(log_nested_dict(v, new_key))
            elif 'runtime' not in k and 'second' not in k:
                filtered_results[new_key] = round(v, 5) if isinstance(v, (float, int)) else v
        return filtered_results

    filtered_results = log_nested_dict(metrics)

    with open(log_path, 'a') as f:
        f.write('\n')
        if header is not None:
            f.write(header + '\n')
        if details is not None:
            for k, v in details.items():
                f.write(f'{k}: {v}\n')
        for k, v in filtered_results.items():
            f.write(f'{k}: {v}\n')
        f.write('\n')


def data_collator(features):
    batch = {key: torch.stack([f[key] for f in features]) for key in features[0]}
    return batch

