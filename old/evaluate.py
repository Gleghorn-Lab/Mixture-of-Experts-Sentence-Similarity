from tqdm.auto import tqdm
from old.utils import log_metrics
from trainer import HF_trainer
from metrics import *


def evaluate_contrastive_model(yargs, model, tokenizer, compute_metrics, get_dataset, data_collator, token):
    training_args = yargs['training_args']
    args = yargs['general_args']
    trainer = HF_trainer(model, train_dataset=None, valid_dataset=None,
                         compute_metrics=compute_metrics, data_collator=data_collator,
                         patience=args['patience'], EX=args['expert_loss'], **training_args)
    
    valid_datasets, test_datasets = get_dataset(args, tokenizer, token) # (aspect, valid_dataset, test_dataset) * aspects * num_dataset
    for i in tqdm(range(len(args['data_paths'])), desc='Evaluating sim metrics'):
        metrics = trainer.evaluate(eval_dataset=valid_datasets[i])
        log_metrics(args['log_path'], metrics, details=args, header=f'Valid dataset {i}')
        metrics = trainer.evaluate(eval_dataset=test_datasets[i])
        log_metrics(args['log_path'], metrics, details=args, header=f'Test dataset {i}')
        trainer.accelerator.free_memory()
