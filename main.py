import argparse
import torch
from trainer import *
from torch.utils.data import DataLoader as TorchLoader
from transformers import BertModel, BertTokenizer
from model import MiniMoELoadWeights, MiniMoE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--data_paths', nargs='+', default=['lhallee/abstract_domain_cvd', 'lhallee/abstract_domain_copd'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--domains', nargs='+', default=['[CVD]', '[COPD]'])
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--validate_interval', type=int, default=10000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patience', type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    base_model = BertModel.from_pretrained(args.model_path)
    base_model.config.attention_probs_dropout_prob = 0.05
    base_model.config.hidden_dropout_prob = 0.05
    loader = MiniMoELoadWeights(base_model=base_model, tokenizer=tokenizer, domains=args.domains)
    model, tokenizer = loader.get_seeded_model()
    mini = MiniMoE(model).to(args.device)

    train_dataset, valid_dataset, test_dataset = get_datasets(args.data_paths, tokenizer, args.domains)
    train_loader = TorchLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = TorchLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = TorchLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(mini.parameters(), lr=args.lr)
    train(args, mini, optimizer, train_loader, valid_loader)
