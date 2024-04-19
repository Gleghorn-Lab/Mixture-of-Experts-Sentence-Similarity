import random
import torch
import numpy as np
import sqlite3
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm


class DoubleDatast(TorchDataset):
    def __init__(self, a, b, c_labels, r_labels, tokenizer, domains, add_tokens, max_length=512):
        self.a = a
        self.b = b
        self.c_labels = c_labels
        self.r_labels = r_labels
        self.tokenizer_base = AutoTokenizer.from_pretrained('lhallee/ankh_base_encoder')
        self.tokenizer = tokenizer
        self.domains = domains
        self.max_length = max_length
        self.add_tokens = add_tokens

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx): # Maybe need a version for non MOE
        r_label = torch.tensor(self.r_labels[idx], dtype=torch.long)
        c_label = torch.tensor(self.c_labels[idx], dtype=torch.float)

        tokenized_a_base = self.tokenizer_base(self.a[idx],
                                     return_tensors='pt',
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_length)
        tokenized_b_base = self.tokenizer_base(self.b[idx],
                                     return_tensors='pt',
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_length)

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
            'base_a_ids':tokenized_a_base['input_ids'].squeeze(),
            'base_b_ids':tokenized_b_base['input_ids'].squeeze(),
            'plm_a_ids':tokenized_a['input_ids'].squeeze(),
            'plm_b_ids':tokenized_b['input_ids'].squeeze(),
            'a_mask':tokenized_a['attention_mask'].squeeze(),
            'b_mask':tokenized_b['attention_mask'].squeeze(),
            'r_labels':r_label,
            'labels':c_label
        }


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
            'pos': p['input_ids'].squeeze(),
            'anc': a['input_ids'].squeeze(),
            'neg': n['input_ids'].squeeze(),
            'att_p': p['attention_mask'].squeeze(),
            'att_a': a['attention_mask'].squeeze(),
            'att_n': n['attention_mask'].squeeze(),
            'r_labels': r_label
        }


class FineTuneDatasetEmbedsFromDisk(TorchDataset):
    def __init__(self, cfg, seqs, labels, task_type='binary'): 
        self.db_file = cfg.db_path
        self.batch_size = cfg.per_device_train_batch_size
        self.emb_dim = cfg.input_dim
        read_scaler = cfg.read_scaler
        self.seqs, self.labels = seqs, labels
        self.length = len(labels)
        self.max_length = len(max(seqs, key=len))
        print('Max length: ', self.max_length)
        self.task_type = task_type
        self.read_amt = read_scaler * self.batch_size
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0
        self.reset_epoch()

    def __len__(self):
        return self.length

    def reset_epoch(self):
        data = list(zip(self.seqs, self.labels))
        random.shuffle(data)
        self.seqs, self.labels = zip(*data)
        self.seqs, self.labels = list(self.seqs), list(self.labels)
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

    def read_embeddings(self):
        embeddings, labels = [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (self.seqs[i],))
            row = result.fetchone()
            emb_data = row[0]
            emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.emb_dim))
            if emb.shape[0] > 1:
                padding_needed = self.max_length - emb.size(0)
                emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed), value=0)
            embeddings.append(emb)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings = embeddings
        self.current_labels = labels

    def __getitem__(self, _):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()
        emb = self.embeddings[self.index]
        label = self.current_labels[self.index]
        self.index += 1
        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)
        return {'embeddings': emb.squeeze(0), 'labels': label}


class FineTuneDatasetEmbeds(TorchDataset):
    def __init__(self, cfg, emb_dict, seqs, labels, task_type='binary'):
        self.embeddings = self.get_embs(emb_dict, seqs)
        self.labels = labels
        self.task_type = task_type
        self.max_length = len(max(seqs, key=len))
        print('Max length: ', self.max_length)
        self.full = cfg.full

    def __len__(self):
        return len(self.labels)
    
    def get_embs(self, emb_dict, seqs):
        embeddings = []
        for seq in tqdm(seqs, desc='Loading Embeddings'):
            emb = emb_dict.get(seq)
            embeddings.append(emb)
        return embeddings

    def __getitem__(self, idx):
        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = torch.tensor(self.embeddings[idx])
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed), value=0)
        return {'embeddings': emb.squeeze(0), 'labels': label}
