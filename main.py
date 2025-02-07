import argparse
import torch
from old.load_model import load_models
from utils import get_yaml
from metrics import compute_metrics_sentence_similarity, compute_metrics_sentence_similarity_test
from evaluate import evaluate_contrastive_model
from data.data_collators import data_collator
from data.get_data import get_datasets_test_sentence_sim
from train import train_sim_model


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--yaml_path', type=str, help='Path to yaml file settings')
    parser.add_argument('--eval', action='store_true', help='Run model in evaluation mode')
    parser.add_argument('--token', type=str, help='Huggingface token')
    return parser.parse_args()


def main():
    ### Set up args
    args = get_args()
    yargs = get_yaml(args.yaml_path)
    for key, value in yargs['general_args'].items(): # copy yaml config into args
        setattr(args, key, value)
    for key, value in yargs['training_args'].items():
        setattr(args, key, value)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### If using wandb
    if args.wandb:
        import wandb
        import os
        os.environ['WANDB_API_KEY'] = input('Wandb api key: ')
        os.environ['WANDB_PROJECT'] = args.wandb_project
        os.environ['WANDB_NAME'] = args.wandb_name
        wandb.login()
        wandb.init()

    print('\n-----Load Model-----\n')
    model, tokenizer = load_models(args)

    if args.eval:
        evaluate_contrastive_model(yargs,
                                   tokenizer=tokenizer,
                                   model=model,
                                   compute_metrics=compute_metrics_sentence_similarity_test,
                                   get_dataset=get_datasets_test_sentence_sim,
                                   data_collator=data_collator,
                                   token=args.token)

    else:
        train_sim_model(yargs,
                        model,
                        tokenizer,
                        compute_metrics=compute_metrics_sentence_similarity,
                        token=args.token)


if __name__ == '__main__':
    main()
