import argparse
import torch

from utils import get_yaml, load_model
from run_model import *
from metrics import compute_metrics_sentence_similarity


def get_args():
    parser = argparse.ArgumentParser(description="MOE settings")
    parser.add_argument('--yaml_path', type=str, )
    parser.add_argument('--eval', action='store_true', help='Run model in evaluation mode.')
    return parser.parse_args()


def main():
    parse = get_args()
    
    yargs = get_yaml(parse.yaml_path)

    args = yargs['general_args']

    if args['wandb']:
        import wandb
        import os
        os.environ['WANDB_API_KEY'] = input('Wandb api key: ')
        os.environ['WANDB_PROJECT'] = args['wandb_project']
        os.environ['WANDB_NAME'] = args['wandb_name']
        wandb.login()
        wandb.init()

    print('\n-----Load Model-----\n')
    model, tokenizer = load_model(args)

    if 'triplet' in args['model_type'].lower():
        compute_metrics = compute_metrics_triplet
    else:
        compute_metrics = compute_metrics_sentence_similarity

    if parse.eval:
        if args['weight_path'] != None:
            weight_path = args['weight_path']
            model.load_state_dict(torch.load(weight_path))
            print(f'Model loaded from {weight_path}')
        evaluate_sim_model(yargs, tokenizer, compute_metrics=compute_metrics, model=model)
    else:
        if args['model_type'] == 'Triplet':
            train_triplet_model(yargs, model, tokenizer, compute_metrics=compute_metrics)
        else:
            train_sim_model(yargs, model, tokenizer, compute_metrics=compute_metrics)


if __name__ == '__main__':
    main()
