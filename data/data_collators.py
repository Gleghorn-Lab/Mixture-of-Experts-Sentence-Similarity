import torch
from typing import List


def get_data_collator(tokenizer, domain_tokens: List[str], max_length: int=512, add_tokens: bool=True):
    def _data_collator(batch):
        a_docs = [item[0] for item in batch]
        b_docs = [item[1] for item in batch]
        expert_assignments = torch.tensor([item[2] for item in batch])

        tokenized_a = tokenizer(
            a_docs,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )

        tokenized_b = tokenizer(
            b_docs,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )

        if add_tokens:
            input_ids_a, input_ids_b = tokenized_a['input_ids'], tokenized_b['input_ids']
            for i in range(len(input_ids_a)):
                domain_token = tokenizer(domain_tokens[int(expert_assignments[i].item())], add_special_tokens=False).input_ids[0]
                input_ids_a[i][0] = domain_token
                input_ids_b[i][0] = domain_token
            tokenized_a['input_ids'], tokenized_b['input_ids'] = input_ids_a, input_ids_b

        return {'a_docs': tokenized_a, 'b_docs': tokenized_b, 'labels': expert_assignments}
    return _data_collator