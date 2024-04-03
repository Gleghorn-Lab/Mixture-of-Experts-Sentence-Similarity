import argparse
import torch
from models.load_model import load_models
from utils import get_yaml
from metrics import compute_metrics_sentence_similarity, compute_metrics_triplet
from evaluate import (
    evaluate_sim_model,
    evaluate_triplet_model_similarity,
    evaluate_triplet_model_downstream,
    evaluate_protein_vec,
    eval_config
)
from train import train_sim_model, train_triplet_model


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--yaml_path', type=str, )
    parser.add_argument('--eval', action='store_true', help='Run model in evaluation mode.')
    parser.add_argument('--token', type=str)
    return parser.parse_args()


def main():
    ### Set up args
    args = get_args()
    yargs = get_yaml(args.yaml_path)
    for key, value in yargs['general_args'].items(): # copy yaml config into args
        setattr(args, key, value)
    for key, value in yargs['training_args'].items():
        setattr(args, key, value)

    ### If using wandb
    if args.wandb:
        import wandb
        import os
        os.environ['WANDB_API_KEY'] = input('Wandb api key: ')
        os.environ['WANDB_PROJECT'] = args.wandb_project
        os.environ['WANDB_NAME'] = args.wandb_name
        wandb.login()
        wandb.init()

    if args.model_type.lower() == 'proteinvec':
        evaluate_protein_vec(yargs)
        import sys
        sys.exit()


    print('\n-----Load Model-----\n')
    model, tokenizer = load_models(args) # if eval and skip, not needed

    if args.weight_path != None:
        if args.huggingface_username in args.weight_path:
            model = model.from_pretrained(args.weight_path, token=args.token)
        else:
            try:
                model.load_state_dict(torch.load(args.weight_path)) # for torch
            except:
                from safetensors.torch import load_model
                load_model(model, args.weight_path) # for safetensors
        print(f'Loaded from {args.weight_path}')

    if args.eval:
        if args.model_type.lower() == 'triplet':
            #evaluate_triplet_model_similarity(yargs, model, tokenizer)
            evaluate_triplet_model_downstream(yargs, eval_config, model, tokenizer)
        else:
            evaluate_sim_model(yargs, tokenizer, model=model)

    else:
        if args.model_type.lower() == 'triplet':
            train_triplet_model(yargs, model, tokenizer, compute_metrics=compute_metrics_triplet, token=args.token)

        else:
            train_sim_model(yargs, model, tokenizer, compute_metrics=compute_metrics_sentence_similarity, token=args.token)


if __name__ == '__main__':
    main()
