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


def create_double_collator(tokenizer_base, tokenizer, max_length):
    def data_collator(features):
        a_texts = [f['plm_a_ids'] for f in features]
        b_texts = [f['plm_b_ids'] for f in features]
        r_labels = torch.stack([f['r_labels'] for f in features])
        c_labels = torch.stack([f['labels'] for f in features])

        tokenized_a_base = tokenizer_base(a_texts, add_special_tokens=True,
                                          return_tensors='pt', padding='longest', truncation=True, max_length=max_length)
        tokenized_b_base = tokenizer_base(b_texts, add_special_tokens=True,
                                          return_tensors='pt', padding='longest', truncation=True, max_length=max_length)
        tokenized_a = tokenizer(a_texts, add_special_tokens=True,
                                return_tensors='pt', padding='longest', truncation=True, max_length=max_length)
        tokenized_b = tokenizer(b_texts, add_special_tokens=True,
                                return_tensors='pt', padding='longest', truncation=True, max_length=max_length)

        plm_a_ids = tokenized_a['input_ids'][:, 1:] # remove cls
        plm_b_ids = tokenized_b['input_ids'][:, 1:]
        a_mask = tokenized_a['attention_mask'][:, 1:]
        b_mask = tokenized_b['attention_mask'][:, 1:]

        batch = {
            'base_a_ids': tokenized_a_base['input_ids'],
            'base_b_ids': tokenized_b_base['input_ids'],
            'plm_a_ids': plm_a_ids,
            'plm_b_ids': plm_b_ids,
            'a_mask': a_mask,
            'b_mask': b_mask,
            'r_labels': r_labels,
            'labels': c_labels
        }

        return batch

    return data_collator
