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
            labels: torch.Tensor,
    ):
        self.a_documents = a_documents
        self.b_documents = b_documents
        self.expert_assignments = expert_assignments
        self.labels = labels

    def __len__(self):
        return len(self.a_documents)

    def __getitem__(self, idx):
        expert_assignment = torch.tensor(self.expert_assignments[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        a_doc, b_doc = self.a_documents[idx], self.b_documents[idx]
        if random.random() < 0.5:
            a_doc, b_doc = b_doc, a_doc
        return a_doc, b_doc, expert_assignment, label


def get_single_eval_data(data_path: str, path_token_dict: Dict[str, str], token_expert_dict: Dict[str, int]) -> SimDataset:
    data = load_dataset(data_path, split='test')
    domain_token = path_token_dict[data_path]
    expert_assignment = token_expert_dict[domain_token]
    expert_assignments = [expert_assignment] * len(data['a'])
    labels = torch.tensor(data['label'])
    return SimDataset(data['a'], data['b'], expert_assignments, labels)


def get_all_eval_data(data_paths: List[str], path_token_dict: Dict[str, str], token_expert_dict: Dict[str, int]) -> SimDataset:
    all_a_documents, all_b_documents, all_expert_assignments, all_labels = [], [], [], []
    for path in data_paths:
        domain_token = path_token_dict[path]
        expert_assignment = token_expert_dict[domain_token]
        data = load_dataset(path, split='test')
        all_a_documents.extend(data['a'])
        all_b_documents.extend(data['b'])
        all_expert_assignments.extend([expert_assignment] * len(data['a']))
        all_labels.extend(data['label'])
    return SimDataset(all_a_documents, all_b_documents, all_expert_assignments, all_labels)



def get_all_eval_documents(data_dict: Dict[str, str], token_expert_dict: Dict[str, int]):
    all_a_documents, all_b_documents, all_domain_tokens, all_labels, all_expert_assignments = [], [], [], [], []
    for domain, data_path in data_dict.items():
        data = load_dataset(data_path, split='test')
        all_a_documents.extend(data['a'])
        all_b_documents.extend(data['b'])
        all_domain_tokens.extend([domain] * len(data['a']))
        all_labels.extend(data['label'])
        expert_assignment = token_expert_dict[domain]
        all_expert_assignments.extend([expert_assignment] * len(data['a']))
    return all_a_documents, all_b_documents, all_domain_tokens, all_labels, all_expert_assignments



def get_single_train_data(
    data_path: str,
    path_token_dict: Dict[str, str],
    token_expert_dict: Dict[str, int],
    cross_validation: bool = False,
    cv: int = 1,
) -> Union[SimDataset, List[SimDataset]]:
    data = load_dataset(data_path, split='train')
    domain_token = path_token_dict[data_path]
    expert_assignment = token_expert_dict[domain_token]

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
            fold_labels = [data['label'][j] for j in fold_indices]

            ds = SimDataset(
                a_documents=fold_a_documents,
                b_documents=fold_b_documents,
                expert_assignments=fold_expert_assignments,
                labels=fold_labels,
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
            labels=data['label'],
        )
        return dataset



def get_all_train_data(
    data_paths: List[str],
    path_token_dict: Dict[str, str],
    token_expert_dict: Dict[str, int],
    cross_validation: bool = False,
    cv: int = 1,
) -> Union[SimDataset, List[SimDataset]]:
    all_a_documents, all_b_documents, all_expert_assignments, all_labels = [], [], [], []
    for path in data_paths:
        domain_token = path_token_dict[path]
        expert_assignment = token_expert_dict[domain_token]
        data = load_dataset(path, split='train')
        all_a_documents.extend(data['a'])
        all_b_documents.extend(data['b'])
        all_expert_assignments.extend([expert_assignment] * len(data['a']))
        all_labels.extend(data['label'])

    random.seed(42)
    entries = list(zip(all_a_documents, all_b_documents, all_expert_assignments))
    random.shuffle(entries)
    # Unzip the entries back to separate lists
    all_a_documents, all_b_documents, all_expert_assignments = zip(*entries)

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
                fold_labels = all_labels[start:]
            else:
                fold_a_documents = all_a_documents[start:start + fold_size]
                fold_b_documents = all_b_documents[start:start + fold_size]
                fold_expert_assignments = all_expert_assignments[start:start + fold_size]
                fold_labels = all_labels[start:start + fold_size]
            ds = SimDataset(
                a_documents=fold_a_documents,
                b_documents=fold_b_documents,
                expert_assignments=fold_expert_assignments,
                labels=fold_labels,
            )

            datasets.append(ds)
        return datasets
    else:
        dataset = SimDataset(
            a_documents=all_a_documents,
            b_documents=all_b_documents,
            expert_assignments=all_expert_assignments,
            labels=all_labels,
        )
        return dataset
