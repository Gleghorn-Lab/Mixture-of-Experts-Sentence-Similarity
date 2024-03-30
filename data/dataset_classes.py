import random
import torch
import numpy as np
import sqlite3
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm


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
        self.emb_dim = cfg.hidden_dim if cfg.full else cfg.input_dim
        read_scaler = cfg.read_scaler
        self.full = cfg.full
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
            if self.full:
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


def embed_dataset(cfg, model, tokenizer, sequences):
    model.eval()
    full, pooling, max_length = cfg.full, cfg.pooling, cfg.max_length
    input_embeddings = []
    with torch.no_grad():
        for sample in tqdm(sequences, desc='Embedding'):
            ids = tokenizer(sample,
                            add_special_tokens=True,
                            padding=False,
                            return_token_type_ids=False,
                            return_tensors='pt').input_ids.to(cfg.device)
            output = model(ids)
            try:
                emb = output.last_hidden_state.float()
            except:
                emb = output.hidden_states[-1].float()

            if full:
                if emb.size(1) < max_length:
                    padding_needed = max_length - emb.size(1)
                    emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed, 0, 0), value=0)
                else:
                    emb = emb[:, :max_length, :]
                input_embeddings.append(emb.detach().cpu().numpy())

            else:
                if pooling == 'cls':
                    emb = emb[:, 0, :]
                elif pooling == 'mean':
                    emb = torch.mean(emb, dim=1, keepdim=False)
                else:
                    emb = torch.max(emb, dim=1, keepdim=False)[0]

                input_embeddings.append(emb.detach().cpu().numpy())
    return input_embeddings


def embed_dataset_and_save(cfg, model, tokenizer, sequences, domains, aspects):
    model.eval()
    db_file = cfg.db_path
    batch_size = 1000

    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc='Batches'):
                batch_sequences = sequences[i:i + batch_size]
                embeddings = []

                for j, sample in tqdm(enumerate(batch_sequences), total=len(batch_sequences), desc='Embedding'):  # Process embeddings in batches
                    ids = tokenizer(sample,
                                    add_special_tokens=True,
                                    padding=False,
                                    return_token_type_ids=False,
                                    return_tensors='pt').input_ids.to(cfg.device)

                    domain_token = tokenizer(domains[aspects[i+j]], add_special_tokens=False).input_ids[0]  # get the domain token
                    ids[0][0] = domain_token  # replace the cls token with the domain token

                    embeddings.append(model.embed(ids).detach().cpu().numpy())

                for seq, emb in zip(batch_sequences, embeddings):
                    emb_data = np.array(emb).tobytes()
                    c.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, emb_data))
                conn.commit()


def save_embeddings_to_disk(emb_dict, db_file="embeddings.db"):
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")

        for seq, emb in emb_dict.items():
            emb_data = np.array(emb).tobytes()  # Serialize using NumPy
            c.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, emb_data))
        conn.commit()


def load_embedding(seq, db_file="embeddings.db"):
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
        row = result.fetchone()
        if row is not None:
            emb_data = row[0]
            return np.frombuffer(emb_data, dtype=np.float32).reshape(-1)  # Deserialize
        else:
            return None