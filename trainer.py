import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset
from tqdm.auto import tqdm

from metrics import *


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


def get_datasets_train(data_paths, tokenizer, domains):
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


def get_datasets_test(data_paths, tokenizer, domains):
    valid_datasets = []
    test_datasets = []
    for i, data_path in enumerate(data_paths):
        valid_a, valid_b, valid_c_label, valid_r_label = [], [], [], []
        test_a, test_b, test_c_label, test_r_label = [], [], [], []
        dataset = load_dataset(data_path)
        valid = dataset['valid']
        test = dataset['test']
        valid_a.extend(valid['a'])
        valid_b.extend(valid['b'])
        valid_c_label.extend(valid['label'])
        valid_r_label.extend([i] * len(valid['label']))
        test_a.extend(test['a'])
        test_b.extend(test['b'])
        test_c_label.extend(test['label'])
        test_r_label.extend([i] * len(test['label']))
        valid_datasets.append(TextDataset(valid_a, valid_b, valid_c_label, valid_r_label, tokenizer, domains))
        test_datasets.append(TextDataset(test_a, test_b, test_c_label, test_r_label, tokenizer, domains))
    return valid_datasets, test_datasets


def test(config, model, test_loader, domain='dataset'):
    model.eval()
    cosine_sims, labels = [], []
    pbar = tqdm(test_loader, total=len(test_loader), desc=f'Evaluating {domain}')
    for (batch1, batch2, c_labels, r_labels) in pbar:
        r_labels = r_labels.to(config.device)
        batch1 = {k:v.squeeze(1).to(config.device) for k, v in batch1.items()}
        batch2 = {k:v.squeeze(1).to(config.device) for k, v in batch2.items()}
        with torch.no_grad():
            try:
                emba, embb, router_logits, c_loss, r_loss = model(batch1, batch2, r_labels)
            except:
                emba, embb, c_loss = model(batch1, batch2, r_labels)
        cosine_sims.extend(F.cosine_similarity(emba, embb).tolist())
        labels.extend(c_labels.tolist())
    cosine_sims_tensor = torch.tensor(cosine_sims, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    threshold, f1max = calc_f1max(cosine_sims_tensor, labels_tensor)
    acc = calc_accuracy(cosine_sims_tensor, labels_tensor, cutoff=threshold)
    dist = calc_distance(cosine_sims_tensor, labels_tensor)
    return threshold, f1max, acc, dist


def validate(config, model, val_loader):
    model.eval()
    cosine_sims, labels = [], []
    pbar = tqdm(val_loader, total=len(val_loader), desc='Validating')
    for (batch1, batch2, c_labels, r_labels) in pbar:
        r_labels = r_labels.to(config.device)
        batch1 = {k:v.squeeze(1).to(config.device) for k, v in batch1.items()}
        batch2 = {k:v.squeeze(1).to(config.device) for k, v in batch2.items()}
        with torch.no_grad():
            try:
                emba, embb, router_logits, c_loss, r_loss = model(batch1, batch2, r_labels)
            except:
                emba, embb, c_loss = model(batch1, batch2, r_labels)
        cosine_sims.extend(F.cosine_similarity(emba, embb).tolist())
        labels.extend(c_labels.tolist())
    cosine_sims_tensor = torch.tensor(cosine_sims, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    threshold, f1max = calc_f1max(cosine_sims_tensor, labels_tensor)
    return threshold, f1max


def train_moebert(config, model, optimizer, train_loader, val_loader, save_path='./best_model.pt'):
    best_val_f1, patience_counter = 0.0, 0
    avg = config.average_interval
    c_losses, r_losses, cos_sims, accuracies = [], [], [], []
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=config.lr,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=config.epochs,
                                                    pct_start=(config.warmup_steps/len(train_loader))/config.epochs) # warmup steps as percentage

    if config.wandb:
        import wandb
        wandb.init(project=config.project_name)
        wandb.watch(model)

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (batch1, batch2, c_labels, r_labels) in pbar:
            r_labels = r_labels.to(config.device)
            batch1 = {k:v.squeeze(1).to(config.device) for k, v in batch1.items()}
            batch2 = {k:v.squeeze(1).to(config.device) for k, v in batch2.items()}
            optimizer.zero_grad()
            emba, embb, router_logits, c_loss, r_loss = model(batch1, batch2, r_labels)
            loss = c_loss + r_loss
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate for warmup

            c_losses.append(c_loss.item())
            r_losses.append(r_loss.item())
            cos_sims.append(F.cosine_similarity(emba, embb).mean().item())
            avg_logits = torch.stack(router_logits, dim=2).transpose(1, 2).mean(dim=1) # batch_size, num_experts
            router_predictions = torch.argmax(avg_logits, dim=1)
            accuracy = (router_predictions == r_labels).float().mean().item()
            accuracies.append(accuracy)
            if len(c_losses) > avg:
                avg_c_loss = np.mean(c_losses[-avg:])
                avg_r_loss = np.mean(r_losses[-avg:])
                avg_cos_sim = np.mean(cos_sims[-avg:])
                avg_accuracy = np.mean(accuracies[-avg:])
                pbar.set_description(f'Epoch {epoch} C_Loss: {avg_c_loss:.4f} R_Loss: {avg_r_loss:.4f} Cosine Similarity: {avg_cos_sim:.4f} Accuracy: {avg_accuracy:.4f}')

                if config.wandb:
                    wandb.log({'C_Loss': avg_c_loss, 'R_Loss': avg_r_loss, 'Cosine Similarity': avg_cos_sim, 'Accuracy': avg_accuracy, 'Learning Rate': scheduler.get_last_lr()[0]})

            if batch_idx % config.validate_interval == 0 and batch_idx > 0:
                threshold, val_f1 = validate(config, model, val_loader)
                model.train()
                print(f'Epoch {epoch} Step {batch_idx} Threshold {threshold} Val F1 ', val_f1)
                if config.wandb:
                    wandb.log({'Threshold': threshold, 'Val F1max': val_f1})
                    if not config.MNR:
                        wandb.log({'Temperature': model.temp.detach().item()})
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter > config.patience:
                        print('Early stopping due to evaluation not improving')
                        model.load_state_dict(torch.load(save_path))
                        return model
    model.load_state_dict(torch.load(save_path))
    return model


def train_bert(config, model, optimizer, train_loader, val_loader, save_path='./best_model.pt'):
    best_val_f1, patience_counter = 0.0, 0
    avg = config.average_interval
    c_losses, cos_sims = [], []
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=config.lr,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=config.epochs,
                                                    pct_start=(config.warmup_steps/len(train_loader))/config.epochs) # warmup steps as percentage

    if config.wandb:
        import wandb
        wandb.init(project=config.project_name)
        wandb.watch(model)

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (batch1, batch2, c_labels, r_labels) in pbar:
            r_labels = r_labels.to(config.device)
            batch1 = {k:v.squeeze(1).to(config.device) for k, v in batch1.items()}
            batch2 = {k:v.squeeze(1).to(config.device) for k, v in batch2.items()}
            optimizer.zero_grad()
            emba, embb, c_loss = model(batch1, batch2, r_labels)
            c_loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate for warmup

            c_losses.append(c_loss.item())
            cos_sims.append(F.cosine_similarity(emba, embb).mean().item())
            if len(c_losses) > avg:
                avg_c_loss = np.mean(c_losses[-avg:])
                avg_cos_sim = np.mean(cos_sims[-avg:])
                pbar.set_description(f'Epoch {epoch} C_Loss: {avg_c_loss:.4f} Cosine Similarity: {avg_cos_sim:.4f}')

                if config.wandb:
                    wandb.log({'C_Loss': avg_c_loss, 'Cosine Similarity': avg_cos_sim, 'Learning Rate': scheduler.get_last_lr()[0]})

            if batch_idx % config.validate_interval == 0 and batch_idx > 0:
                threshold, val_f1 = validate(config, model, val_loader)
                model.train()
                print(f'Epoch {epoch} Step {batch_idx} Threshold {threshold} Val F1 ', val_f1)
                if config.wandb:
                    wandb.log({'Threshold': threshold, 'Val F1max': val_f1})
                    if not config.MNR:
                        wandb.log({'Temperature': model.temp.detach().item()})
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter > config.patience:
                        print('Early stopping due to evaluation not improving')
                        model.load_state_dict(torch.load(save_path))
                        return model
    model.load_state_dict(torch.load(save_path))
    print('Early stopping not reached. Training concluded.')
    return model
