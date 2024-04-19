import argparse
import torch
from models.load_model import load_models
from utils import get_yaml, create_double_collator
from utils import data_collator as standard_data_collator
from metrics import (
    compute_metrics_sentence_similarity,
    compute_metrics_sentence_similarity_test,
    compute_metrics_double,
    compute_metrics_triplet
)
from evaluate import (
    evaluate_contrastive_model,
    evaluate_model_downstream,
    evaluate_protein_vec,
    eval_config
)
from data.load_data import (
    get_datasets_test_sentence_sim,
    get_datasets_test_triplet
)
from train import train_sim_model, train_triplet_model, train_double_model


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
        evaluate_protein_vec(yargs, token=args.token)
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
            evaluate_contrastive_model(yargs,
                                       tokenizer=tokenizer,
                                       model=model,
                                       compute_metrics=compute_metrics_triplet,
                                       get_dataset=get_datasets_test_triplet,
                                       data_collator=standard_data_collator,
                                       token=args.token)
            evaluate_model_downstream(yargs, eval_config, model, tokenizer, token=args.token)
        elif args.model_type.lower() == 'sentencesimilarity':
            evaluate_contrastive_model(yargs,
                                       tokenizer=tokenizer,
                                       model=model,
                                       compute_metrics=compute_metrics_sentence_similarity_test,
                                       get_dataset=get_datasets_test_sentence_sim,
                                       data_collator=standard_data_collator,
                                       token=args.token)
        else:
            from transformers import AutoTokenizer
            max_length = args.max_length
            tokenizer_base = AutoTokenizer.from_pretrained('lhallee/ankh_base_encoder')
            double_collator = create_double_collator(tokenizer_base, tokenizer, max_length)
            evaluate_contrastive_model(yargs,
                                       tokenizer=tokenizer,
                                       model=model,
                                       compute_metrics=compute_metrics_double,
                                       get_dataset=get_datasets_test_sentence_sim,
                                       data_collator=double_collator,
                                       token=args.token)
            evaluate_model_downstream(yargs, eval_config, model, tokenizer, token=args.token)
    else:
        if args.model_type.lower() == 'triplet':
            train_triplet_model(yargs, model, tokenizer,
                                compute_metrics=compute_metrics_triplet, token=args.token)

        elif args.model_type.lower() == 'sentencesimilarity':
            train_sim_model(yargs, model, tokenizer,
                            compute_metrics=compute_metrics_sentence_similarity, token=args.token)

        else:
            train_double_model(yargs, model, tokenizer,
                               compute_metrics=compute_metrics_double, token=args.token)

if __name__ == '__main__':
    main()
