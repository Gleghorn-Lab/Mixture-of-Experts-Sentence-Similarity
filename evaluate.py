from data.load_data import (
    embed_dataset_and_save,
    get_fine_tune_data,
    get_seqs,
    get_datasets_test_sentence_sim,
)
from data.dataset_classes import FineTuneDatasetEmbedsFromDisk
from models.model_zoo import LinearClassifier
from utils import log_metrics, data_collator
from trainer import HF_trainer
from metrics import *

# Holder class
class eval_config:
    db_path = 'embeddings.db'


def evaluate_sim_model(yargs, tokenizer, compute_metrics, model=None, trainer=None):
    training_args = yargs['training_args']
    args = yargs['general_args']
    data_paths = args['data_paths']

    if args['domains'] is not None and args['add_during_eval']:
        added_tokens = {'additional_special_tokens' : args['domains']}
        tokenizer.add_special_tokens(added_tokens)
        print(tokenizer)
    
    validation_datasets, testing_datasets = get_datasets_test_sentence_sim(args, tokenizer)
    details = {
        'model_path': args['model_path'],
        'MOE': args['MOE']
    }
    
    if trainer == None:
        trainer = HF_trainer(model, validation_datasets[0], testing_datasets[0],
                        compute_metrics=compute_metrics, data_collator=data_collator, **training_args)

    for i, val_dataset in enumerate(validation_datasets):
        metrics = trainer.predict(val_dataset)[-1]
        log_metrics(args['log_path'], metrics, details=details, header=f'Validation {data_paths[i]}')

    for i, test_dataset in enumerate(testing_datasets):
        metrics = trainer.predict(test_dataset)[-1]
        log_metrics(args['log_path'], metrics, details=details, header=f'Test {data_paths[i]}')


def evaluate_triplet_model(yargs, eval_config, base_model, tokenizer): # TODO add PPI and SSQ
    training_args = yargs['eval_training_args']
    eval_args = yargs['eval_args']
    general_args = yargs['general_args']

    # Set up config
    for key, value in eval_args.items():
        setattr(eval_config, key, value)
    for key, value in training_args:
        setattr(eval_config, key, value)
    args = eval_config
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets, all_seqs, train_sets, valid_sets, test_sets, num_labels, task_types = [], [], [], [], [], [], []
    seq_to_origin = {}

    # Get all datasets
    for i, data_path in enumerate(args.data_paths):
        train_set, valid_set, test_set, num_label, task_type = get_fine_tune_data(args, data_path)
        num_labels.append(num_label)
        task_types.append(task_type)

        train_seqs, train_labels = get_seqs(train_set)
        valid_seqs, valid_labels = get_seqs(valid_set)
        test_seqs, test_labels = get_seqs(test_set)

        train_seqs, train_labels = train_seqs[:10], train_labels[:10]
        valid_seqs, valid_labels = valid_seqs[:10], valid_labels[:10]
        test_seqs, test_labels = test_seqs[:10], test_labels[:10]

        train_sets.append((train_seqs, train_labels))
        valid_sets.append((valid_seqs, valid_labels))
        test_sets.append((test_seqs, test_labels))

        for seq in train_seqs + valid_seqs + test_seqs:
            seq_to_origin[seq] = i

        all_seqs.extend(train_seqs + valid_seqs + test_seqs)

    all_seqs = list(set(all_seqs))
    aspects = [seq_to_origin[seq] for seq in all_seqs]

    # If not already done, embed all seqs to local sql database
    if not args.skip:
        embed_dataset_and_save(args, base_model, tokenizer, all_seqs, domains=args.domains, aspects=aspects)
        base_model.to('cpu')
        del base_model

    # Make datasets
    for i in range(len(train_sets)):
        train_dataset = FineTuneDatasetEmbedsFromDisk(args, train_sets[i][0], train_sets[i][1], task_types[i])
        valid_dataset = FineTuneDatasetEmbedsFromDisk(args, valid_sets[i][0], valid_sets[i][1], task_types[i])
        test_dataset = FineTuneDatasetEmbedsFromDisk(args, test_sets[i][0], test_sets[i][1], task_types[i])
        datasets.append((train_dataset, valid_dataset, test_dataset))

    # For each dataset, train a model and record test results
    for i, dataset in enumerate(datasets):
        print(f'Training {args.data_paths[i]}, {i+1} / {len(datasets)}')
        train_dataset, valid_dataset, test_dataset = dataset
        task_type, num_label = task_types[i], num_labels[i]
        
        model = LinearClassifier(args, task_type=task_type, num_labels=num_label)
        
        if task_type == 'singlelabel':
            compute_metrics = compute_metrics_single_label_classification
        elif task_type == 'multilabel':
            compute_metrics = compute_metrics_multi_label_classification
        else:
            compute_metrics = compute_metrics_regression

        trainer = HF_trainer(model, train_dataset, valid_dataset,
                             compute_metrics=compute_metrics, data_collator=data_collator,
                             patience=args['patience'], MI=False, **training_args)
        trainer.train()

        metrics = trainer.evaluate(test_dataset)
        trainer.accelerator.free_memory()
        ### TODO write one set of details first and then all metrics
        log_metrics(general_args['log_path'], metrics, details=general_args, header=f'Evaluation {args.data_paths[i]}')
