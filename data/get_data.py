import random
import torch
from typing import Any, List, Dict, Union
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset


DATA_DICT = {
    '[COPD]': 'GleghornLab/abstract_domain_copd',
    '[CVD]': 'GleghornLab/abstract_domain_cvd',
    '[CANCER]': 'GleghornLab/abstract_domain_skincancer',
    '[PARASITIC]': 'GleghornLab/abstract_domain_parasitic',
    '[AUTOIMMUNE]': 'GleghornLab/abstract_domain_autoimmune',
}

path_token_dict = {
    'GleghornLab/abstract_domain_copd': '[COPD]',
    'GleghornLab/abstract_domain_cvd': '[CVD]',
    'GleghornLab/abstract_domain_skincancer': '[CANCER]',
    'GleghornLab/abstract_domain_parasitic': '[PARASITIC]',
    'GleghornLab/abstract_domain_autoimmune': '[AUTOIMMUNE]'
}

token_expert_dict = {
    '[COPD]': 0,
    '[CVD]': 1,
    '[CANCER]': 2,
    '[PARASITIC]': 3,
    '[AUTOIMMUNE]': 4
}


class SimDataset(TorchDataset):
    def __init__(
            self,
            a_documents: List[str],
            b_documents: List[str],
            expert_assignments: torch.Tensor,
            domain_tokens: List[str],
            tokenizer: Any,
            add_tokens: bool = True,
            max_length: int = 512
    ):
        self.a_documents = a_documents
        self.b_documents = b_documents
        self.expert_assignments = expert_assignments
        self.domain_tokens = domain_tokens
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_tokens = add_tokens

    def __len__(self):
        return len(self.a_documents)

    def __getitem__(self, idx):
        expert_assignment = torch.tensor(self.expert_assignments[idx], dtype=torch.long)
        a_doc, b_doc = self.a_documents[idx], self.b_documents[idx]
        if random.random() < 0.5:
            a_doc, b_doc = b_doc, a_doc
        tokenized_a = self.tokenizer(
            a_doc,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )

        tokenized_b = self.tokenizer(
            b_doc,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )

        if self.add_tokens:
            domain_token = self.tokenizer(self.domain_tokens[int(expert_assignment.item())],
                                          add_special_tokens=False).input_ids[0]  # get the domain token

            tokenized_a['input_ids'][0][0] = domain_token  # replace the cls token with the domain token
            tokenized_b['input_ids'][0][0] = domain_token

        return {
            'a_doc': tokenized_a,
            'b_doc': tokenized_b,
            'labels': expert_assignment, # we call label instead of assignment to play nice with compute_metrics
        }


def get_single_eval_data(
        data_path: str,
        tokenizer: Any,
        domain_token_dict: Dict[str, str],
        token_expert_dict: Dict[str, int],
        max_length: int = 1024,
        add_tokens: bool = True,

) -> SimDataset:
    data = load_dataset(data_path, split='test')
    domain_label = token_expert_dict[data_path]
    domain_token = domain_token_dict[domain_label]
    domain_labels = [domain_label] * len(data['a'])
    domains = [domain_token] * len(data['a'])
    return SimDataset(data['a'], data['b'], domain_labels, domains, tokenizer, add_tokens=add_tokens, max_length=max_length)


def get_all_eval_documents(data_dict: Dict[str, str]):
    all_a_documents, all_b_documents, all_domain_tokens, all_labels = [], [], [], []
    for domain, data_path in data_dict.items():
        data = load_dataset(data_path, split='valid')
        all_a_documents.extend(data['a'])
        all_b_documents.extend(data['b'])
        all_domain_tokens.extend([domain] * len(data['a']))
        all_labels.extend(data['label'])
    return all_a_documents, all_b_documents, all_domain_tokens, all_labels


def get_single_train_data(
    data_path: str,
    tokenizer: Any,
    path_token_dict: Dict[str, str],
    token_expert_dict: Dict[str, int],
    max_length: int = 512,
    add_tokens: bool = True,
    cross_validation: bool = False,
    cv: int = 1,
) -> Union[SimDataset, List[SimDataset]]:
    data = load_dataset(data_path, split='train')
    domain_token = path_token_dict[data_path]
    expert_assignment = token_expert_dict[domain_token]
    domain_tokens = list(path_token_dict.values())


    if cross_validation and cv > 1:
        # Create a list of shuffled indices for splitting
        indices = list(range(len(data['a'])))
        random.shuffle(indices)
        fold_size = len(indices) // cv
        datasets = []
        for i in range(cv):
            start = i * fold_size
            # Make sure the last fold takes any remainder
            if i == cv - 1:
                fold_indices = indices[start:]
            else:
                fold_indices = indices[start:start + fold_size]
            fold_a_documents = [data['a'][j] for j in fold_indices]
            fold_b_documents = [data['b'][j] for j in fold_indices]
            fold_expert_assignments = [expert_assignment] * len(fold_indices)
            ds = SimDataset(
                a_documents=fold_a_documents,
                b_documents=fold_b_documents,
                expert_assignments=fold_expert_assignments,
                domain_tokens=domain_tokens,
                tokenizer=tokenizer,
                max_length=max_length,
                add_tokens=add_tokens,
            )
            datasets.append(ds)
        return datasets
    else:
        # No cross validation splitting, return full dataset
        expert_assignments = [expert_assignment] * len(data['a'])
        dataset = SimDataset(
            a_documents=data['a'],
            b_documents=data['b'],
            expert_assignments=expert_assignments,
            domain_tokens=domain_tokens,
            tokenizer=tokenizer,
            max_length=max_length,
            add_tokens=add_tokens,
        )
        return dataset


def get_all_train_data(
    data_paths: List[str],
    tokenizer: Any,
    path_token_dict: Dict[str, str],
    token_expert_dict: Dict[str, int],
    max_length: int = 512,
    add_tokens: bool = True,
    cross_validation: bool = False,
    cv: int = 1,
) -> Union[SimDataset, List[SimDataset]]:
    all_a_documents, all_b_documents, all_expert_assignments = [], [], []
    for path in data_paths:
        domain_token = path_token_dict[path]
        expert_assignment = token_expert_dict[domain_token]
        # Here we select a subset (first 100 entries) from the train split
        data = load_dataset(path, split='train').select(range(100))
        all_a_documents.extend(data['a'])
        all_b_documents.extend(data['b'])
        all_expert_assignments.extend([expert_assignment] * len(data['a']))

    random.seed(42)
    entries = list(zip(all_a_documents, all_b_documents, all_expert_assignments))
    random.shuffle(entries)
    # Unzip the entries back to separate lists
    all_a_documents, all_b_documents, all_expert_assignments = zip(*entries)
    domain_tokens = list(path_token_dict.values())

    if cross_validation and cv > 1:
        total_entries = len(all_a_documents)
        fold_size = total_entries // cv
        datasets = []
        for i in range(cv):
            start = i * fold_size
            # Last fold takes any remaining entries
            if i == cv - 1:
                fold_a_documents = all_a_documents[start:]
                fold_b_documents = all_b_documents[start:]
                fold_expert_assignments = all_expert_assignments[start:]
            else:
                fold_a_documents = all_a_documents[start:start + fold_size]
                fold_b_documents = all_b_documents[start:start + fold_size]
                fold_expert_assignments = all_expert_assignments[start:start + fold_size]
            ds = SimDataset(
                a_documents=fold_a_documents,
                b_documents=fold_b_documents,
                expert_assignments=fold_expert_assignments,
                domain_tokens=domain_tokens,
                tokenizer=tokenizer,
                max_length=max_length,
                add_tokens=add_tokens,
            )
            datasets.append(ds)
        return datasets
    else:
        dataset = SimDataset(
            a_documents=all_a_documents,
            b_documents=all_b_documents,
            expert_assignments=all_expert_assignments,
            domain_tokens=domain_tokens,
            tokenizer=tokenizer,
            max_length=max_length,
            add_tokens=add_tokens,
        )
        return dataset
