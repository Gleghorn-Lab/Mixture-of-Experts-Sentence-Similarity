import torch
from tqdm.auto import tqdm
from data.load_data import (
    get_datasets_train_sentence_sim,
    get_datasets_test_triplet,
    get_datasets_train_triplet
)
from trainer import HF_trainer
from evaluate import (
    evaluate_sim_model,
    evaluate_triplet_model_downstream,
    eval_config
)
from utils import log_metrics, data_collator


def train_sim_model(yargs, model, tokenizer, compute_metrics, token=None):
    training_args = yargs['training_args']
    args = yargs['general_args']
    train_dataset, valid_dataset = get_datasets_train_sentence_sim(args, tokenizer)[:2]

    trainer = HF_trainer(model, train_dataset, valid_dataset,
                         compute_metrics=compute_metrics, data_collator=data_collator,
                         patience=args['patience'], EX=args['expert_loss'], **training_args)
    trainer.train()

    save_path = args['save_path']
    if save_path is not None:
        torch.save(trainer.model, f=save_path)
        hub_path = args['huggingface_username'] + '/' + save_path.split('.')[0]
        trainer.model.push_to_hub(hub_path, token=token, private=True)
        tokenizer.push_to_hub(hub_path, toke=token, private=True)
        print(f'Model saved at {save_path} and pushed to {hub_path}')

    trainer.accelerator.free_memory()
    evaluate_sim_model(yargs, tokenizer, model=trainer.model)


def train_triplet_model(yargs, model, tokenizer, compute_metrics, token=None):
    training_args = yargs['training_args']
    args = yargs['general_args']
    train_dataset, valid_dataset = get_datasets_train_triplet(args, tokenizer)[:2]

    trainer = HF_trainer(model, train_dataset, valid_dataset,
                         compute_metrics=compute_metrics, data_collator=data_collator,
                         patience=args['patience'], EX=args['expert_loss'], **training_args)
    trainer.train()

    save_path = args['save_path']
    if save_path is not None:
        torch.save(trainer.model, f=save_path)
        hub_path = args['huggingface_username'] + '/' + save_path.split('.')[0]
        trainer.model.push_to_hub(hub_path, token=token, private=True)
        tokenizer.push_to_hub(hub_path, toke=token, private=True)
        print(f'Model saved at {save_path} and pushed to {hub_path}')
    
    triplet_datasets = get_datasets_test_triplet(args, tokenizer) # (aspect, valid_dataset, test_dataset) * aspects * num_dataset
    for (aspect, valid_dataset, test_dataset) in tqdm(triplet_datasets, desc='Evaluating sim metrics'):
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        log_metrics(args['log_path'], metrics, details=args, header=f'Valid aspect {aspect}')
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        log_metrics(args['log_path'], metrics, details=args, header=f'Test aspect {aspect}')

    trainer.accelerator.free_memory()
    evaluate_triplet_model_downstream(yargs, eval_config=eval_config, 
                                      ase_model=trainer.model, tokenizer=tokenizer)
