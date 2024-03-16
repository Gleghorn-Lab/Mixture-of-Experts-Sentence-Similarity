import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset


def data_collator(features):
    batch = {key: torch.stack([f[key] for f in features]) for key in features[0]}
    return batch


class TextDataset(TorchDataset):
    def __init__(self, a, b, c_labels, r_labels, tokenizer, domains, max_length=512):
        self.a = a
        self.b = b
        self.c_labels = c_labels
        self.r_labels = r_labels
        self.tokenizer = tokenizer
        self.domains = domains
        self.max_length = max_length

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx): # Maybe need a version for non MOE
        r_label = torch.tensor(self.r_labels[idx], dtype=torch.long)
        c_label = torch.tensor(self.c_labels[idx], dtype=torch.float)
        domain_token = self.tokenizer(self.domains[int(r_label.item())], add_special_tokens=False).input_ids[0]  # get the domain token
        tokenized_a = self.tokenizer(self.a[idx],
                                     return_tensors='pt',
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_length)
        tokenized_b = self.tokenizer(self.b[idx],
                                     return_tensors='pt',
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_length)
        tokenized_a['input_ids'][0][0] = domain_token  # replace the cls token with the domain token
        tokenized_b['input_ids'][0][0] = domain_token  # replace the cls token with the domain token
        return {
            'input_ids_a': tokenized_a['input_ids'].squeeze(),
            'attention_mask_a': tokenized_a['attention_mask'].squeeze(),
            'input_ids_b': tokenized_b['input_ids'].squeeze(),
            'attention_mask_b': tokenized_b['attention_mask'].squeeze(),
            'labels': c_label,
            'r_labels': r_label
        }


def get_datasets_train(data_paths, tokenizer, domains, max_length=512):
    train_a, train_b, train_c_label, train_r_label = [], [], [], []
    valid_a, valid_b, valid_c_label, valid_r_label = [], [], [], []
    test_a, test_b, test_c_label, test_r_label = [], [], [], []
    for i, data_path in enumerate(data_paths):
        dataset = load_dataset(data_path)
        train = dataset['train']
        valid = dataset['valid']
        test = dataset['test']
        train_a.extend(train['a'])
        train_b.extend(train['b'])
        train_c_label.extend(train['label'])
        train_r_label.extend([i] * len(train['label']))
        valid_a.extend(valid['a'])
        valid_b.extend(valid['b'])
        valid_c_label.extend(valid['label'])
        valid_r_label.extend([i] * len(valid['label']))
        test_a.extend(test['a'])
        test_b.extend(test['b'])
        test_c_label.extend(test['label'])
        test_r_label.extend([i] * len(test['label']))
    train_dataset = TextDataset(train_a, train_b, train_c_label, train_r_label,
                                tokenizer, domains, max_length)
    valid_dataset = TextDataset(valid_a, valid_b, valid_c_label, valid_r_label,
                                tokenizer, domains, max_length)
    test_dataset = TextDataset(test_a, test_b, test_c_label, test_r_label,
                               tokenizer, domains, max_length)
    return train_dataset, valid_dataset, test_dataset


def get_datasets_test(data_paths, tokenizer, domains, max_length=512):
    valid_datasets = []
    test_datasets = []
    for i, data_path in enumerate(data_paths):
        valid_a, valid_b, valid_c_label, valid_r_label = [], [], [], []
        test_a, test_b, test_c_label, test_r_label = [], [], [], []
        dataset = load_dataset(data_path)
        valid = dataset['valid']
        test = dataset['test']
        valid_a.extend(valid['a'])
        valid_b.extend(valid['b'])
        valid_c_label.extend(valid['label'])
        valid_r_label.extend([i] * len(valid['label']))
        test_a.extend(test['a'])
        test_b.extend(test['b'])
        test_c_label.extend(test['label'])
        test_r_label.extend([i] * len(test['label']))
        valid_datasets.append(TextDataset(valid_a, valid_b, valid_c_label, valid_r_label,
                                          tokenizer, domains, max_length))
        test_datasets.append(TextDataset(test_a, test_b, test_c_label, test_r_label,
                                         tokenizer, domains, max_length))
    return valid_datasets, test_datasets
