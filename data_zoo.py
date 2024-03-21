import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset


def data_collator(features):
    batch = {key: torch.stack([f[key] for f in features]) for key in features[0]}
    return batch


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


class TripletDataset(TorchDataset):
    def __init__(self, positives, anchors, negatives,
                 r_labels, tokenizer, domains, add_tokens, max_length=512):
        self.positives = positives
        self.anchors = anchors
        self.negatives = negatives
        self.r_labels = r_labels
        self.tokenizer = tokenizer
        self.domains = domains
        self.max_length = max_length
        self.add_tokens = add_tokens

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        r_label = torch.tensor(self.r_labels[idx], dtype=torch.long)

        p = self.tokenizer(self.positives[idx],
                           return_tensors='pt',
                           padding='max_length',
                           truncation=True,
                           max_length=self.max_length)
        
        a = self.tokenizer(self.anchors[idx],
                           return_tensors='pt',
                           padding='max_length',
                           truncation=True,
                           max_length=self.max_length)
        
        n = self.tokenizer(self.negatives[idx],
                           return_tensors='pt',
                           padding='max_length',
                           truncation=True,
                           max_length=self.max_length)

        if self.add_tokens:
            domain_token = self.tokenizer(self.domains[int(r_label.item())],
                                          add_special_tokens=False).input_ids[0]  # get the domain token
            p['input_ids'][0][0] = domain_token  # replace the cls token with the domain token
            a['input_ids'][0][0] = domain_token
            n['input_ids'][0][0] = domain_token

        return {
            'p': p['input_ids'].squeeze(),
            'a': a['input_ids'].squeeze(),
            'n': n['input_ids'].squeeze(),
            'att_p': p['attention_mask'].squeeze(),
            'att_a': a['attention_mask'].squeeze(),
            'att_n': n['attention_mask'].squeeze(),
            'r_labels': r_label
        }


def get_datasets_train_sentence_sim(args, tokenizer):
    data_paths = args['data_paths']
    domains = args['domains']
    add_tokens = args['new_special_tokens']
    max_length = args['max_length']
    a_col = args['a_col']
    b_col = args['b_col']
    label_col = args['label_col']

    train_a, train_b, train_c_label, train_r_label = [], [], [], []
    valid_a, valid_b, valid_c_label, valid_r_label = [], [], [], []
    test_a, test_b, test_c_label, test_r_label = [], [], [], []
    for i, data_path in enumerate(data_paths):
        dataset = load_dataset(data_path)
        train = dataset['train']
        valid = dataset['valid']
        test = dataset['test']
        train_a.extend(train[a_col])
        train_b.extend(train[b_col])
        train_c_label.extend(train[label_col])
        train_r_label.extend([i] * len(train[label_col]))
        valid_a.extend(valid[a_col])
        valid_b.extend(valid[b_col])
        valid_c_label.extend(valid[label_col])
        valid_r_label.extend([i] * len(valid[label_col]))
        test_a.extend(test[a_col])
        test_b.extend(test[b_col])
        test_c_label.extend(test[label_col])
        test_r_label.extend([i] * len(test[label_col]))
    train_dataset = SimDataset(train_a, train_b, train_c_label, train_r_label,
                                tokenizer, domains, add_tokens, max_length)
    valid_dataset = SimDataset(valid_a, valid_b, valid_c_label, valid_r_label,
                                tokenizer, domains, add_tokens,  max_length)
    test_dataset = SimDataset(test_a, test_b, test_c_label, test_r_label,
                               tokenizer, domains, add_tokens,  max_length)
    return train_dataset, valid_dataset, test_dataset


def get_datasets_test_sentence_sim(args, tokenizer):
    data_paths = args['data_paths']
    domains = args['domains']
    add_tokens = args['new_special_tokens']
    max_length = args['max_length']
    a_col = args['a_col']
    b_col = args['b_col']
    label_col = args['label_col']

    valid_datasets = []
    test_datasets = []
    for i, data_path in enumerate(data_paths):
        valid_a, valid_b, valid_c_label, valid_r_label = [], [], [], []
        test_a, test_b, test_c_label, test_r_label = [], [], [], []
        dataset = load_dataset(data_path)
        valid = dataset['valid']
        test = dataset['test']
        valid_a.extend(valid[a_col])
        valid_b.extend(valid[b_col])
        valid_c_label.extend(valid[label_col])
        valid_r_label.extend([i] * len(valid[label_col]))
        test_a.extend(test[a_col])
        test_b.extend(test[b_col])
        test_c_label.extend(test[label_col])
        test_r_label.extend([i] * len(test[label_col]))
        valid_datasets.append(SimDataset(valid_a, valid_b, valid_c_label, valid_r_label,
                                          tokenizer, domains, add_tokens, max_length))
        test_datasets.append(SimDataset(test_a, test_b, test_c_label, test_r_label,
                                         tokenizer, domains, add_tokens, max_length))
    return valid_datasets, test_datasets


def get_datasets_train_triplet(args, tokenizer):
    data_paths = args['data_paths']
    domains = args['domains']
    add_tokens = args['new_special_tokens']
    max_length = args['max_length']
    p_col = 'positives'
    a_col = 'anchors'
    n_col = 'negatives'
    label_col = 'aspects'

    valid_size = args['valid_size']
    test_size = args['test_size']

    train_p, train_a, train_n, train_label = [], [], [], []
    valid_p, valid_a, valid_n, valid_label = [], [], [], []
    test_p, test_a, test_n, test_label = [], [], [], []

    for i, data_path in enumerate(data_paths):
        dataset = load_dataset(data_path)
        train = dataset['train']
        valid = dataset['valid']
        test = dataset['test']

        train_p.extend(train[p_col])
        train_a.extend(train[a_col])
        train_n.extend(train[n_col])
        train_label.extend(train[label_col])

        valid_p.extend(valid[p_col][:valid_size])
        valid_a.extend(valid[a_col][:valid_size])
        valid_n.extend(valid[n_col][:valid_size])
        valid_label.extend(label_col)

        test_p.extend(test[p_col][:test_size])
        test_a.extend(test[a_col][:test_size])
        test_n.extend(test[n_col][:test_size])
        test_label.extend(label_col)

    train_dataset = TripletDataset(train_p, train_a, train_n, train_label,
                                   tokenizer, domains, add_tokens, max_length)
    valid_dataset = TripletDataset(valid_p, valid_a, valid_n, valid_label,
                                   tokenizer, domains, add_tokens, max_length)
    test_dataset = TripletDataset(test_p, test_a, test_n, test_label,
                                  tokenizer, domains, add_tokens, max_length)

    return train_dataset, valid_dataset, test_dataset


def get_datasets_test_triplet(args, tokenizer):
    data_paths = args['data_paths']
    domains = args['domains']
    add_tokens = args['new_special_tokens']
    max_length = args['max_length']
    p_col = 'positives'
    a_col = 'anchors'
    n_col = 'negatives'
    label_col = 'aspects'

    valid_datasets = []
    test_datasets = []

    for i, data_path in enumerate(data_paths):
        valid_p, valid_a, valid_n, valid_label = [], [], [], []
        test_p, test_a, test_n, test_label = [], [], [], []

        dataset = load_dataset(data_path)
        valid = dataset['valid']
        test = dataset['test']

        valid_p.extend(valid[p_col])
        valid_a.extend(valid[a_col])
        valid_n.extend(valid[n_col])
        valid_label.extend(label_col)

        test_p.extend(test[p_col])
        test_a.extend(test[a_col])
        test_n.extend(test[n_col])
        test_label.extend(label_col)

        valid_datasets.append(TripletDataset(valid_p, valid_a, valid_n, valid_label,
                                             tokenizer, domains, add_tokens, max_length))
        test_datasets.append(TripletDataset(test_p, test_a, test_n, test_label,
                                            tokenizer, domains, add_tokens, max_length))

    return valid_datasets, test_datasets
