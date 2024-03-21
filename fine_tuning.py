import random
import torch
import numpy as np
import sqlite3
from torch import nn
from datasets import load_dataset
from torch.utils import data
from tqdm.auto import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput


def get_loss_fct(task_type):
    if task_type == 'singlelabel':
        loss_fct = nn.CrossEntropyLoss()
    elif task_type == 'multilabel':
        loss_fct = nn.BCEWithLogitsLoss()
    elif task_type == 'regression':
        loss_fct = nn.MSELoss()
    else:
        print(f'Specified wrong classification type {task_type}')
    return loss_fct


class LinearClassifier(nn.Module):
    def __init__(self, cfg, task_type='binary', num_labels=2):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.num_layers = cfg.num_layers
        self.input_layer = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            self.hidden_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.output_layer = nn.Linear(cfg.hidden_dim, num_labels)
        self.loss_fct = get_loss_fct(task_type)

    def forward(self, embeddings, labels=None):
        embeddings = self.gelu(self.input_layer(embeddings))
        for i in range(self.num_layers):
            embeddings = self.dropout(self.gelu(self.hidden_layers[i](embeddings)))
        logits = self.output_layer(embeddings)
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


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


def embed_dataset_and_save(cfg, model, tokenizer, sequences, aspect=0):
    model.eval()
    db_file = cfg.db_path
    full, pooling, max_length = cfg.full, cfg.pooling, cfg.max_length
    batch_size = 1000

    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc='Batches'):
                batch_sequences = sequences[i:i + batch_size]
                embeddings = []

                for sample in tqdm(batch_sequences, desc='Embedding'):  # Process embeddings in batches
                    ids = tokenizer(sample,
                                    add_special_tokens=True,
                                    padding=False,
                                    return_token_type_ids=False,
                                    return_tensors='pt').input_ids.to(cfg.device)

                    ### ADD DOMAIN TOKENS ###

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


class FineTuneDatasetEmbedsFromDisk(data.Dataset):
    def __init__(self, cfg, seqs, labels, task_type='binary'): 
        self.db_file = cfg.db_path
        self.batch_size = cfg.batch_size
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

    def __getitem__(self, idx):
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


def get_data(cfg, data_path):
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
