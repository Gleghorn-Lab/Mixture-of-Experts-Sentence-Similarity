import argparse
import torch
from models.load_model import load_models
from utils import get_yaml
from metrics import compute_metrics_sentence_similarity, compute_metrics_triplet
from evaluate import evaluate_sim_model, evaluate_triplet_model, eval_config
from train import train_sim_model, train_triplet_model


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--yaml_path', type=str, )
    parser.add_argument('--eval', action='store_true', help='Run model in evaluation mode.')
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


    print('\n-----Load Model-----\n')
    model, tokenizer = load_models(args)


    if 'triplet' in args.model_type.lower():
        compute_metrics = compute_metrics_triplet
    else:
        compute_metrics = compute_metrics_sentence_similarity


    if args.eval: ### TODO make this robust for all options
        if args.huggingface_username in args.weight_path:
            model = model.from_pretrained(args.weight_path, token=args.token)
        else:
            try:
                model.load_state_dict(torch.load(args.weight_path)) # for torch
            except:
                from safetensors.torch import load_model
                load_model(model, args.weight_path) # for safetensors
        print(f'Loaded from {args.weight_path}')

        if args.model_type == 'Triplet':
            evaluate_triplet_model(yargs, eval_config, model, tokenizer)

        else:
            evaluate_sim_model(yargs, tokenizer, compute_metrics=compute_metrics, model=model)

    else:

        if args.model_type == 'Triplet':
            train_triplet_model(yargs, model, tokenizer, compute_metrics=compute_metrics)

        else:
            train_sim_model(yargs, model, tokenizer, compute_metrics=compute_metrics)


if __name__ == '__main__':
    main()
