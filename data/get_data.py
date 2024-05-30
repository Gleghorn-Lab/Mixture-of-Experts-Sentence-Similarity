import random
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset


class SimDataset(TorchDataset):
    def __init__(self, a, b, c_labels, r_labels, tokenizer, domains, add_tokens, max_length=512):
        self.a = a
        self.b = b
        self.c_labels = c_labels
        self.r_labels = r_labels
        self.tokenizer = tokenizer
        self.domains = domains
        self.max_length = max_length
        self.add_tokens = add_tokens

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx): # Maybe need a version for non MOE
        r_label = torch.tensor(self.r_labels[idx], dtype=torch.long)
        c_label = torch.tensor(self.c_labels[idx], dtype=torch.float)
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
        if self.add_tokens:
            domain_token = self.tokenizer(self.domains[int(r_label.item())],
                                          add_special_tokens=False).input_ids[0]  # get the domain token
            tokenized_a['input_ids'][0][0] = domain_token  # replace the cls token with the domain token
            tokenized_b['input_ids'][0][0] = domain_token
        return {
            'a': tokenized_a['input_ids'].squeeze(),
            'b': tokenized_b['input_ids'].squeeze(),
            'att_a': tokenized_a['attention_mask'].squeeze(),
            'att_b': tokenized_b['attention_mask'].squeeze(),
            'labels': c_label,
            'r_labels': r_label
        }


def get_datasets_train_sentence_sim(args, tokenizer, token):
    data_paths = args['data_paths']
    domains = args['domains']
    add_tokens = args['new_special_tokens']
    max_length = args['max_length']
    a_col = args['a_col']
    b_col = args['b_col']
    label_col = args['label_col']
    valid_size = args['valid_size']
    test_size = args['test_size']
    DATASET = SimDataset

    train_a, train_b, train_c_label, train_r_label = [], [], [], []
    valid_a, valid_b, valid_c_label, valid_r_label = [], [], [], []
    test_a, test_b, test_c_label, test_r_label = [], [], [], []
    for i, data_path in enumerate(data_paths):
        dataset = load_dataset(data_path, token=token)
        train = dataset['train']
        valid = dataset['valid']
        test = dataset['test']
        train_a.extend(train[a_col])
        train_b.extend(train[b_col])
        if label_col in train.column_names:
            train_c_label.extend(train[label_col])
        else:
            train_c_label.extend([0] * len(train[a_col]))
        train_r_label.extend([i] * len(train[a_col]))
        valid_a.extend(valid[a_col])
        valid_b.extend(valid[b_col])
        if label_col in valid.column_names:
            valid_c_label.extend(valid[label_col])
        else:
            valid_c_label.extend([0] * len(valid[a_col]))
        valid_r_label.extend([i] * len(valid[a_col]))
        test_a.extend(test[a_col])
        test_b.extend(test[b_col])
        if label_col in test.column_names:
            test_c_label.extend(test[label_col])
        else:
            test_c_label.extend([0] * len(test[a_col]))
        test_r_label.extend([i] * len(test[a_col]))

    if len(valid) > valid_size:
        valid = list(zip(valid_a[:valid_size], valid_b[:valid_size], valid_r_label[:valid_size], valid_c_label[:valid_size]))
        random.shuffle(valid)
        valid_a, valid_b, valid_r_label, valid_c_label = zip(*valid)

    if len(test) > test_size:
        test = list(zip(test_a[:test_size], test_b[:test_size], test_r_label[:test_size], test_c_label[:test_size]))
        random.shuffle(test)
        test_a, test_b, test_r_label, test_c_label = zip(*test)

    train_dataset = DATASET(train_a, train_b, train_c_label, train_r_label,
                                tokenizer, domains, add_tokens, max_length)
    valid_dataset = DATASET(valid_a, valid_b, valid_c_label, valid_r_label,
                                tokenizer, domains, add_tokens,  max_length)
    test_dataset = DATASET(test_a, test_b, test_c_label, test_r_label,
                               tokenizer, domains, add_tokens,  max_length)
    return train_dataset, valid_dataset, test_dataset


def get_datasets_test_sentence_sim(args, tokenizer, token):
    data_paths = args['data_paths']
    domains = args['domains']
    add_tokens = args['new_special_tokens']
    max_length = args['max_length']
    a_col = args['a_col']
    b_col = args['b_col']
    label_col = args['label_col']
    DATASET = SimDataset

    valid_datasets = []
    test_datasets = []
    for i, data_path in enumerate(data_paths):
        valid_a, valid_b, valid_c_label, valid_r_label = [], [], [], []
        test_a, test_b, test_c_label, test_r_label = [], [], [], []
        dataset = load_dataset(data_path, token=token)
        valid = dataset['valid']
        test = dataset['test']
        valid_a.extend(valid[a_col])
        valid_b.extend(valid[b_col])
        if label_col in valid.column_names:
            valid_c_label.extend(valid[label_col])
        else:
            valid_c_label.extend([0] * len(valid[a_col]))
        valid_r_label.extend([i] * len(valid[a_col]))
        test_a.extend(test[a_col])
        test_b.extend(test[b_col])
        if label_col in test.column_names:
            test_c_label.extend(test[label_col])
        else:
            test_c_label.extend([0] * len(test[a_col]))
        test_r_label.extend([i] * len(test[a_col]))
        valid_datasets.append(DATASET(valid_a, valid_b, valid_c_label, valid_r_label,
                                          tokenizer, domains, add_tokens, max_length))
        test_datasets.append(DATASET(test_a, test_b, test_c_label, test_r_label,
                                         tokenizer, domains, add_tokens, max_length))
    return valid_datasets, test_datasets
