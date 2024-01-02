import argparse
import torch
from trainer import *
from torch.utils.data import DataLoader as TorchLoader
from transformers import BertModel, BertTokenizer
from model import MiniMoELoadWeights, MiniMoE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='MiniMOE')
    parser.add_argument('--model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--data_paths', nargs='+', default=['lhallee/abstract_domain_cvd', 'lhallee/abstract_domain_copd'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--domains', nargs='+', default=['[CVD]', '[COPD]'])
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--validate_interval', type=int, default=1000)
    parser.add_argument('--average_interval', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--specific', action='store_true')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--c_scale', type=float, default=1.0)
    parser.add_argument('--r_scale', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    print('\n-----Config-----\n')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    base_model = BertModel.from_pretrained(args.model_path)
    base_model.config.attention_probs_dropout_prob = args.dropout
    base_model.config.hidden_dropout_prob = args.dropout
    loader = MiniMoELoadWeights(base_model=base_model, tokenizer=tokenizer, domains=args.domains)
    model, tokenizer = loader.get_seeded_model()
    mini = MiniMoE(model, specific=args.specific, c_scale=args.c_scale, r_scale=args.r_scale).to(args.device)

    train_dataset, valid_dataset, test_dataset = get_datasets(args.data_paths, tokenizer, args.domains)
    train_loader = TorchLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = TorchLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = TorchLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(mini.parameters(), lr=args.lr)
    train(args, mini, optimizer, train_loader, valid_loader)
