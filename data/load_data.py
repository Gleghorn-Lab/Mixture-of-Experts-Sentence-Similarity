import re
from datasets import load_dataset
from .dataset_classes import *


def not_regression(labels): # not a great assumption but works most of the time
    return all(isinstance(label, (int, float)) and label == int(label) for label in labels)


def label_type_checker(labels):
    ex = labels[0]
    if not_regression(labels):
        if isinstance(ex, list):
            label_type = 'multilabel'
        elif isinstance(ex, int) or isinstance(ex, float):
            label_type = 'singlelabel' # binary or multiclass
    elif isinstance(ex, str):
        label_type = 'string'
    else:
        label_type = 'regression'
    return label_type


def get_seqs(dataset, seq_col='seqs', label_col='labels'):
    return dataset[seq_col], dataset[label_col]


def get_fine_tune_data(cfg, data_path):
    dataset = load_dataset(data_path)
    train_set, valid_set, test_set = dataset['train'], dataset['valid'], dataset['test']

    if cfg.trim:
        original_train_size, original_valid_size, original_test_size = len(train_set), len(valid_set), len(test_set)
        train_set = train_set.filter(lambda x: len(x['seqs'].replace(' ', '')) <= cfg.max_length)
        valid_set = valid_set.filter(lambda x: len(x['seqs'].replace(' ', '')) <= cfg.max_length)
        test_set = test_set.filter(lambda x: len(x['seqs'].replace(' ', '')) <= cfg.max_length)
        print(f'Trimmed {round((original_train_size-len(train_set))/original_train_size, 2)}% from train')
        print(f'Trimmed {round((original_valid_size-len(valid_set))/original_valid_size, 2)}% from valid')
        print(f'Trimmed {round((original_test_size-len(test_set))/original_test_size, 2)}% from test')
    
    check_labels = valid_set['labels']
    label_type = label_type_checker(check_labels)

    if label_type == 'string':
        import ast
        train_set = train_set.map(lambda example: {'labels': ast.literal_eval(example['labels'])})
        valid_set = valid_set.map(lambda example: {'labels': ast.literal_eval(example['labels'])})
        test_set = test_set.map(lambda example: {'labels': ast.literal_eval(example['labels'])})
        label_type = 'multilabel'
    try:
        num_labels = len(train_set['labels'][0])
    except:
        num_labels = len(np.unique(train_set['labels']))
    if label_type == 'regression':
        num_labels = 1
    return train_set, valid_set, test_set, num_labels, label_type


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
    p_col = args['p_col']
    a_col = args['a_col']
    n_col = args['n_col']
    label_col = args['label_col']
    valid_size = args['valid_size']
    test_size = args['test_size']

    train_p, train_a, train_n, train_label = [], [], [], []
    valid_p, valid_a, valid_n, valid_label = [], [], [], []
    test_p, test_a, test_n, test_label = [], [], [], []

    for data_path in data_paths:
        dataset = load_dataset(data_path)
        train = dataset['train']
        valid = dataset['valid']
        test = dataset['test']

        train_p.extend(train[p_col])
        train_a.extend(train[a_col])
        train_n.extend(train[n_col])
        train_label.extend(train[label_col])

        valid_p.extend(valid[p_col])
        valid_a.extend(valid[a_col])
        valid_n.extend(valid[n_col])
        valid_label.extend(valid[label_col])

        test_p.extend(test[p_col])
        test_a.extend(test[a_col])
        test_n.extend(test[n_col])
        test_label.extend(test[label_col])
    
    valid = list(zip(valid_p[:valid_size], valid_a[:valid_size], valid_n[:valid_size], valid_label[:valid_size]))
    random.shuffle(valid)
    valid_p, valid_a, valid_n, valid_label = zip(*valid)

    test = list(zip(test_p[:test_size], test_a[:test_size], test_n[:test_size], test_label[:test_size]))
    random.shuffle(test)
    test_p, test_a, test_n, test_label = zip(*test)


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
    p_col = args['p_col']
    a_col = args['a_col']
    n_col = args['n_col']
    label_col = args['label_col']

    datasets_by_label = {}

    for data_path in data_paths:
        dataset = load_dataset(data_path)
        valid = dataset['valid']
        test = dataset['test']

        for label in set(valid[label_col]):
            if label not in datasets_by_label:
                datasets_by_label[label] = {
                    'valid_p': [], 'valid_a': [], 'valid_n': [], 'valid_label': [],
                    'test_p': [], 'test_a': [], 'test_n': [], 'test_label': []
                }

            label_mask_valid = [l == label for l in valid[label_col]]
            label_mask_test = [l == label for l in test[label_col]]

            datasets_by_label[label]['valid_p'].extend([p for p, m in zip(valid[p_col], label_mask_valid) if m])
            datasets_by_label[label]['valid_a'].extend([a for a, m in zip(valid[a_col], label_mask_valid) if m])
            datasets_by_label[label]['valid_n'].extend([n for n, m in zip(valid[n_col], label_mask_valid) if m])
            datasets_by_label[label]['valid_label'].extend([l for l, m in zip(valid[label_col], label_mask_valid) if m])

            datasets_by_label[label]['test_p'].extend([p for p, m in zip(test[p_col], label_mask_test) if m])
            datasets_by_label[label]['test_a'].extend([a for a, m in zip(test[a_col], label_mask_test) if m])
            datasets_by_label[label]['test_n'].extend([n for n, m in zip(test[n_col], label_mask_test) if m])
            datasets_by_label[label]['test_label'].extend([l for l, m in zip(test[label_col], label_mask_test) if m])

    if args['model_type'] == 'ProteinVec':
        for label in datasets_by_label:
            datasets_by_label[label]['valid_p'] = [(" ".join(sequence)) for sequence in datasets_by_label[label]['valid_p']]
            datasets_by_label[label]['valid_p'] = [re.sub(r"[UZOB]", "X", sequence) for sequence in datasets_by_label[label]['valid_p']]
            datasets_by_label[label]['valid_a'] = [(" ".join(sequence)) for sequence in datasets_by_label[label]['valid_a']]
            datasets_by_label[label]['valid_a'] = [re.sub(r"[UZOB]", "X", sequence) for sequence in datasets_by_label[label]['valid_a']]
            datasets_by_label[label]['valid_n'] = [(" ".join(sequence)) for sequence in datasets_by_label[label]['valid_n']]
            datasets_by_label[label]['valid_n'] = [re.sub(r"[UZOB]", "X", sequence) for sequence in datasets_by_label[label]['valid_n']]

            datasets_by_label[label]['test_p'] = [(" ".join(sequence)) for sequence in datasets_by_label[label]['test_p']]
            datasets_by_label[label]['test_p'] = [re.sub(r"[UZOB]", "X", sequence) for sequence in datasets_by_label[label]['test_p']]
            datasets_by_label[label]['test_a'] = [(" ".join(sequence)) for sequence in datasets_by_label[label]['test_a']]
            datasets_by_label[label]['test_a'] = [re.sub(r"[UZOB]", "X", sequence) for sequence in datasets_by_label[label]['test_a']]
            datasets_by_label[label]['test_n'] = [(" ".join(sequence)) for sequence in datasets_by_label[label]['test_n']]
            datasets_by_label[label]['test_n'] = [re.sub(r"[UZOB]", "X", sequence) for sequence in datasets_by_label[label]['test_n']]

    triplet_datasets = []
    for label, dataset_data in datasets_by_label.items():
        valid_dataset = TripletDataset(dataset_data['valid_p'], dataset_data['valid_a'], dataset_data['valid_n'],
                                    dataset_data['valid_label'], tokenizer, domains, add_tokens, max_length)
        test_dataset = TripletDataset(dataset_data['test_p'], dataset_data['test_a'], dataset_data['test_n'],
                                    dataset_data['test_label'], tokenizer, domains, add_tokens, max_length)
        triplet_datasets.append((label, valid_dataset, test_dataset))

    return triplet_datasets
