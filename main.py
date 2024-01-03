import argparse
import torch
from trainer import *
from torch.utils.data import DataLoader as TorchLoader
from transformers import BertModel, BertTokenizer
from model import BertForSentenceSimilarity, BertMoEForSentenceSimilarity, BertMoELoadWeights


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='MiniMOE')
    parser.add_argument('--model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--data_paths', nargs='+', default=['lhallee/abstract_domain_cvd', 'lhallee/abstract_domain_copd'])
    parser.add_argument('--log_path', type=str, default='./results.txt')
    parser.add_argument('--save_path', type=str, default='./best_model.pt')
    parser.add_argument('--weight_path', type=str, default='./best_model.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--domains', nargs='+', default=['[CVD]', '[COPD]'])
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--validate_interval', type=int, default=5000)
    parser.add_argument('--average_interval', type=int, default=25)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--MNR', action='store_true')
    parser.add_argument('--specific', action='store_true')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--MOE', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--c_scale', type=float, default=1.0)
    parser.add_argument('--r_scale', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()


def evaluate_model(args, trained_model, tokenizer):
    validation_datasets, testing_datasets = get_datasets_test(args.data_paths, tokenizer, args.domains)
    with open(args.log_path, 'w') as f:
        for i, val_dataset in enumerate(validation_datasets):
            val_loader = TorchLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            threshold, f1max, acc, dist = test(args, trained_model, val_loader, args.domain[i])
            f.write(f'\n-----Validation Metrics {args.domains[i]}-----\n')
            f.write(f'Threshold: {threshold}\n')
            f.write(f'F1 Max: {f1max}\n')
            f.write(f'Accuracy: {acc}\n')
            f.write(f'Distance: {dist}\n')

        for i, test_dataset in enumerate(testing_datasets):
            test_loader = TorchLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            threshold, f1max, acc, dist = test(args, trained_model, test_loader, args.domain[i])
            f.write(f'\n-----Testing Metrics {args.domains[i]}-----\n')
            f.write(f'Threshold: {threshold}\n')
            f.write(f'F1 Max: {f1max}\n')
            f.write(f'Accuracy: {acc}\n')
            f.write(f'Distance: {dist}\n')


def train_model(args, model, tokenizer):
    train_dataset, valid_dataset, _ = get_datasets_train(args.data_paths, tokenizer, args.domains)
    train_loader = TorchLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = TorchLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.MOE:
        trained_model = train_moebert(args, model, optimizer, train_loader, valid_loader, args.save_path)
    else:
        trained_model = train_bert(args, model, optimizer, train_loader, valid_loader, args.save_path)
    print(f'Model saved at {args.save_path}')
    evaluate_model(args, trained_model, tokenizer)


def load_model(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    base_model = BertModel.from_pretrained(args.model_path)
    base_model.config.attention_probs_dropout_prob = args.dropout
    base_model.config.hidden_dropout_prob = args.dropout
    if args.MOE:
        loader = BertMoELoadWeights(base_model=base_model, tokenizer=tokenizer, domains=args.domains)
        base_model, tokenizer = loader.get_seeded_model()
        model = BertMoEForSentenceSimilarity(base_model, MNR=args.MNR, specific=args.specific, c_scale=args.c_scale, r_scale=args.r_scale).to(args.device)
    else:
        model = BertForSentenceSimilarity(base_model, args.MNR, args.c_scale).to(args.device)
    return model, tokenizer


if __name__ == '__main__':
    args = get_args()
    
    print('\n-----Config-----\n')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    
    print('\n-----Load Model-----\n')
    model, tokenizer = load_model(args)
    if args.eval:
        model.load_state_dict(torch.load(args.weight_path))
        print(f'Model loaded from {args.weight_path}')
        evaluate_model(args, model, tokenizer)
    else:
        train_model(args, model, tokenizer)