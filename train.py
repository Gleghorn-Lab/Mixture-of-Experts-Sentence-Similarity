import torch
from data.get_data import get_datasets_train_sentence_sim, get_datasets_test_sentence_sim
from data.data_collators import data_collator
from trainer import HF_trainer
from evaluate import evaluate_contrastive_model


def train_sim_model(yargs, model, tokenizer, compute_metrics, token=None):
    training_args = yargs['training_args']
    args = yargs['general_args']
    train_dataset, valid_dataset = get_datasets_train_sentence_sim(args, tokenizer, token)[:2]

    trainer = HF_trainer(model, train_dataset, valid_dataset,
                         compute_metrics=compute_metrics, data_collator=data_collator,
                         patience=args['patience'], EX=args['expert_loss'], token=token, **training_args)
    trainer.train()

    save_path = args['save_path']
    if save_path is not None:
        torch.save(trainer.model, f=save_path)
        hub_path = args['huggingface_username'] + '/' + save_path.split('.')[0]
        trainer.model.push_to_hub(hub_path, token=token, private=True)
        tokenizer.push_to_hub(hub_path, toke=token, private=True)
        print(f'Model saved at {save_path} and pushed to {hub_path}')

    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()
    evaluate_contrastive_model(yargs,
                               tokenizer,
                               model=trainer.model,
                               compute_metrics=compute_metrics,
                               get_dataset=get_datasets_test_sentence_sim)
