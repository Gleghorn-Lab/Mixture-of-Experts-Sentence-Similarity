import random
import torch
from typing import Any, List, Dict
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset


DATA_DICT = {
    '[COPD]': 'GleghornLab/abstract_domain_copd',
    '[CVD]': 'GleghornLab/abstract_domain_cvd',
    '[CANCER]': 'GleghornLab/abstract_domain_skincancer',
    '[PARASITIC]': 'GleghornLab/abstract_domain_parasitic',
    '[AUTOIMMUNE]': 'GleghornLab/abstract_domain_autoimmune',
}


class SimDataset(TorchDataset):
    def __init__(
            self,
            a_documents: List[str],
            b_documents: List[str],
            domain_labels: torch.Tensor,
            domains: List[str],
            tokenizer: Any,
            add_tokens: bool = True,
            max_length: int = 1024

    ):
        self.a_documents = a_documents
        self.b_documents = b_documents
        self.domain_labels = domain_labels
        self.domains = domains
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_tokens = add_tokens

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        domain_label = torch.tensor(self.domain_labels[idx], dtype=torch.long)
        a_doc, b_doc = self.a_documents[idx], self.b_documents[idx]
        if random.random() < 0.5:
            a_doc, b_doc = b_doc, a_doc
        tokenized_a = self.tokenizer(a_doc,
                                     return_tensors='pt',
                                     padding='longest',
                                     truncation=True,
                                     max_length=self.max_length)

        tokenized_b = self.tokenizer(b_doc,
                                     return_tensors='pt',
                                     padding='longest',
                                     truncation=True,
                                     max_length=self.max_length)
        if self.add_tokens:
            domain_token = self.tokenizer(self.domains[int(domain_label.item())],
                                          add_special_tokens=False).input_ids[0]  # get the domain token
            tokenized_a['input_ids'][0][0] = domain_token  # replace the cls token with the domain token
            tokenized_b['input_ids'][0][0] = domain_token

        return {
            'a_doc': a_doc,
            'b_doc': b_doc,
            'labels': domain_label, # we call label instead of assignment to play nice with compute_metrics
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
    return SimDataset(data['a'], data['b'], domain_labels, domains, tokenizer, max_length, add_tokens)


def get_all_eval_documents(data_dict: Dict[str, str]):
    all_a_documents, all_b_documents, all_domain_tokens, all_labels = [], [], [], []
    for domain, data_path in data_dict.items():
        data = load_dataset(data_path, split='test')
        all_a_documents.extend(data['a'])
        all_b_documents.extend(data['b'])
        all_domain_tokens.extend([domain] * len(data['a']))
        all_labels.extend(data['label'])
    return all_a_documents, all_b_documents, all_domain_tokens, all_labels



def get_single_train_data(
    data_path: str,
    tokenizer: Any,
    domain_token_dict: Dict[str, str],
    token_expert_dict: Dict[str, int],
    max_length: int = 1024,
    add_tokens: bool = True,
) -> SimDataset:
    data = load_dataset(data_path, split='train')
    domain_label = token_expert_dict[data_path]
    domain_token = domain_token_dict[domain_label]
    domain_labels = [domain_label] * len(data['a'])
    domains = [domain_token] * len(data['a'])
    return SimDataset(data['a'], data['b'], domain_labels, domains, tokenizer, max_length, add_tokens)


def get_all_train_data(
    data_paths: List[str],
    tokenizer: Any,
    domain_token_dict: Dict[str, str],
    token_expert_dict: Dict[str, int],
    max_length: int = 1024,
    add_tokens: bool = True,
):
    all_a_documents, all_b_documents, all_domain_labels, all_domains = [], [], [], []
    for path in data_paths:
        domain_label = token_expert_dict[path]
        domain_token = domain_token_dict[domain_label]
        data = load_dataset(path, split='train')
        all_a_documents.extend(data['a'])
        all_b_documents.extend(data['b'])
        all_domain_labels.extend([domain_label] * len(data['a']))
        all_domains.extend([domain_token] * len(data['a']))
    return SimDataset(all_a_documents, all_b_documents, all_domain_labels, all_domains, tokenizer, max_length, add_tokens)

