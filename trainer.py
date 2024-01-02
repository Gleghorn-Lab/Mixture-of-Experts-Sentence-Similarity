import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset
from tqdm.auto import tqdm

from metrics import calc_f1max


class TextDataset(TorchDataset):
    def __init__(self, a, b, c_labels, r_labels, tokenizer, domains):
        self.a = a
        self.b = b
        self.c_labels = c_labels
        self.r_labels = r_labels
        self.tokenizer = tokenizer
        self.domains = domains

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        r_label = torch.tensor(self.r_labels[idx], dtype=torch.long)
        c_label = torch.tensor(self.c_labels[idx], dtype=torch.float)
        domain_token = self.tokenizer(self.domains[int(r_label.item())], add_special_tokens=False).input_ids[0]  # get the domain token
        tokenized_a = self.tokenizer(self.a[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        tokenized_b = self.tokenizer(self.b[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        tokenized_a['input_ids'][0][0] = domain_token  # replace the cls token with the domain token
        tokenized_b['input_ids'][0][0] = domain_token  # replace the cls token with the domain token
        return tokenized_a, tokenized_b, c_label, r_label


def get_datasets(data_paths, tokenizer, domains):
    train_a, train_b, train_c_label, train_r_label = [], [], [], []
    valid_a, valid_b, valid_c_label, valid_r_label = [], [], [], []
    test_a, test_b, test_c_label, test_r_label = [], [], [], []
    for i, data_path in enumerate(data_paths):
        dataset = load_dataset(data_path)
        train = dataset['train']
        valid = dataset['valid']
        test = dataset['test']
        train_a.extend(train['a'])
        train_b.extend(train['b'])
        train_c_label.extend(train['label'])
        train_r_label.extend([i] * len(train['label']))
        valid_a.extend(valid['a'])
        valid_b.extend(valid['b'])
        valid_c_label.extend(valid['label'])
        valid_r_label.extend([i] * len(valid['label']))
        test_a.extend(test['a'])
        test_b.extend(test['b'])
        test_c_label.extend(test['label'])
        test_r_label.extend([i] * len(test['label']))
    train_dataset = TextDataset(train_a, train_b, train_c_label, train_r_label, tokenizer, domains)
    valid_dataset = TextDataset(valid_a, valid_b, valid_c_label, valid_r_label, tokenizer, domains)
    test_dataset = TextDataset(test_a, test_b, test_c_label, test_r_label, tokenizer, domains)
    return train_dataset, valid_dataset, test_dataset


def validate(config, model, val_loader):
    model.eval()
    cosine_sims, labels = [], []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for batch_idx, (batch1, batch2, c_labels, r_labels) in pbar:
        r_labels = r_labels.to(config.device)
        batch1 = {k:v.squeeze(1).to(config.device) for k, v in batch1.items()}
        batch2 = {k:v.squeeze(1).to(config.device) for k, v in batch2.items()}
        with torch.no_grad():
            emba, embb, loss = model(batch1, batch2, r_labels)
        cosine_sims.extend(F.cosine_similarity(emba, embb).tolist())
        labels.extend(c_labels.tolist())
    cosine_sims_tensor = torch.tensor(cosine_sims, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    f1max = calc_f1max(cosine_sims_tensor, labels_tensor)
    return f1max


def train(config, model, optimizer, train_loader, val_loader):
    best_val_f1 = float('inf')
    patience_counter = 0
    losses, cos_sims = [], []

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (batch1, batch2, c_labels, r_labels) in pbar:
            r_labels = r_labels.to(config.device)
            batch1 = {k:v.squeeze(1).to(config.device) for k, v in batch1.items()}
            batch2 = {k:v.squeeze(1).to(config.device) for k, v in batch2.items()}
            optimizer.zero_grad()
            emba, embb, loss = model(batch1, batch2, r_labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            cos_sims.append(F.cosine_similarity(emba, embb).mean().item())

            if len(losses) > 100:
                avg_loss = np.mean(losses[-100:])
                avg_cos_sim = np.mean(cos_sims[-100:])
                pbar.set_description(f'Epoch {epoch} Loss: {avg_loss:.4f} Cosine Similarity: {avg_cos_sim:.4f}')

            if batch_idx % config.validate_interval == 0:
                threshold, val_f1 = validate(config, model, val_loader)
                print(f'Epoch {epoch} Step {batch_idx} Threshold {threshold} Val F1 ', val_f1)
                if val_f1 < best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    torch.save(model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter > config.patience:
                        print('Early stopping due to loss not improving')
                        model.load_state_dict(torch.load('best_model.pt'))
                        return model
    model.load_state_dict(torch.load('best_model.pt'))
    return model
