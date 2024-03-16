from utils import log_metrics
from trainer import HF_trainer
from data_zoo import *


def evaluate_model(yargs, tokenizer, compute_metrics, model=None, trainer=None):
    training_args = yargs['training_args']
    args = yargs['general_args']
    data_paths = args['data_paths']
    validation_datasets, testing_datasets = get_datasets_test(data_paths, tokenizer, args['domains'], args['max_length'])
    details = {
        'model_path': args['model_path'],
        'MOE': args['MOE']
    }
    
    if trainer == None:
        trainer = HF_trainer(model, validation_datasets[0], testing_datasets[0],
                        compute_metrics=compute_metrics, data_collator=data_collator, **training_args)

    for i, val_dataset in enumerate(validation_datasets):
        metrics = trainer.predict(val_dataset)[-1]
        log_metrics(args.log_path, metrics, details=details, header=f'Validation {data_paths[i]}')

    for i, test_dataset in enumerate(testing_datasets):
        metrics = trainer.predict(test_dataset)[-1]
        log_metrics(args.log_path, metrics, details=details, header=f'Test {data_paths[i]}')


def train_model(yargs, model, tokenizer, compute_metrics):
    training_args = yargs['training_args']
    args = yargs['general_args']
    train_dataset, valid_dataset = get_datasets_train(args['data_paths'], tokenizer, args['domains'], args['max_length'])[:2]

    trainer = HF_trainer(model, train_dataset, valid_dataset,
                         compute_metrics=compute_metrics, data_collator=data_collator, patience=args['patience'], **training_args)
    trainer.train()
    weight_path = args['weight_path']
    torch.save(trainer.model, f=weight_path)
    print(f'Model saved at {weight_path}')

    evaluate_model(yargs, tokenizer, trainer=trainer, compute_metrics=compute_metrics)