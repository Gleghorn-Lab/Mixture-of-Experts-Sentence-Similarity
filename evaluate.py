from tqdm.auto import tqdm
from data.load_data import (
    get_datasets_test_triplet,
    get_fine_tune_data,
    get_seqs,
)
from data.dataset_classes import FineTuneDatasetEmbedsFromDisk, FineTuneDatasetEmbeds
from data.embed_datasets import *
from models.model_zoo import LinearClassifier, TokenClassifier
from utils import log_metrics
from utils import data_collator as standard_data_collator
from trainer import HF_trainer
from metrics import *


# Holder class
class eval_config:
    db_path = 'embeddings.db'


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


def train_downstream_model(args,
                           training_args,
                           general_args,
                           task_types,
                           num_labels,
                           dataset, i):
    print(f'Training {args.data_paths[i]}, {i+1} / {len(args.data_paths)}')
    train_dataset, valid_dataset, test_dataset = dataset
    task_type, num_label = task_types[i], num_labels[i]

    if task_type == 'tokenwise':
        model = TokenClassifier(args, num_labels=num_label)
    else:
        model = LinearClassifier(args, task_type=task_type, num_labels=num_label)
    
    if task_type == 'singlelabel' or task_type == 'tokenwise':
        compute_metrics = compute_metrics_single_label_classification
    elif task_type == 'multilabel':
        compute_metrics = compute_metrics_multi_label_classification
    else:
        compute_metrics = compute_metrics_regression

    trainer = HF_trainer(model, train_dataset=train_dataset, valid_dataset=valid_dataset,
                            compute_metrics=compute_metrics, data_collator=standard_data_collator,
                            patience=args.patience, EX=False, **training_args)
    trainer.train()

    metrics = trainer.evaluate(test_dataset)
    trainer.accelerator.free_memory()
    ### TODO write one set of details first and then all metrics
    log_metrics(general_args['log_path'], metrics, details=general_args, header=f'Evaluation {args.data_paths[i]}')


def evaluate_model_downstream(yargs, eval_config, base_model, tokenizer, token): # TODO add PPI and SSQ
    training_args = yargs['eval_training_args']
    eval_args = yargs['eval_args']
    general_args = yargs['general_args']
    train_domains = general_args['domains']

    for key, value in general_args.items(): # Set up config
        setattr(eval_config, key, value)
    for key, value in eval_args.items():
        setattr(eval_config, key, value)
    for key, value in training_args.items():
        setattr(eval_config, key, value)
    args = eval_config
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = args.model_type
    domain_dict = {train_domains[i]:i for i in range(len(train_domains))}
    eval_domains = args.domains

    all_seqs, train_sets, valid_sets, test_sets, num_labels, task_types = [], [], [], [], [], []

    for i, data_path in enumerate(args.data_paths): # Get all datasets
        train_set, valid_set, test_set, num_label, task_type = get_fine_tune_data(args, data_path, token)
        num_labels.append(num_label)
        task_types.append(task_type)

        train_seqs, train_labels = get_seqs(train_set)
        valid_seqs, valid_labels = get_seqs(valid_set)
        test_seqs, test_labels = get_seqs(test_set)

        train_seqs, train_labels = train_seqs, train_labels
        valid_seqs, valid_labels = valid_seqs, valid_labels
        test_seqs, test_labels = test_seqs, test_labels

        train_sets.append((train_seqs, train_labels))
        valid_sets.append((valid_seqs, valid_labels))
        test_sets.append((test_seqs, test_labels))

        if args.sql and not args.skip:
            all_seqs.extend(train_seqs + valid_seqs + test_seqs)

    if args.sql:
        if not args.skip:
            all_seqs = list(set(all_seqs)) # set of all sequences from all datasets
            base_model.to(args.device)
            embed_dataset_and_save(args, base_model, tokenizer, all_seqs)
            base_model.to('cpu')
            del base_model

        for i in range(len(train_sets)):
            train_dataset = FineTuneDatasetEmbedsFromDisk(args, train_sets[i][0], train_sets[i][1], task_types[i])
            valid_dataset = FineTuneDatasetEmbedsFromDisk(args, valid_sets[i][0], valid_sets[i][1], task_types[i])
            test_dataset = FineTuneDatasetEmbedsFromDisk(args, test_sets[i][0], test_sets[i][1], task_types[i])
            dataset = (train_dataset, valid_dataset, test_dataset)
            train_downstream_model(args, training_args, general_args, task_types, num_labels, dataset, i)

    else:
        for i in range(len(train_sets)):
            base_model.to(args.device)
            seqs = train_sets[i][0] + valid_sets[i][0] + test_sets[i][0]
            if model_type.lower() == 'triplet':
                expert = domain_dict[eval_domains[i]]  # int for what expert to call based on original train domains
                emb_dict = dict(zip(seqs, embed_domain_moe_dataset(args, base_model, tokenizer, seqs, expert, eval_domains[i])))
            elif model_type.lower() == 'proteinvec':
                aspect_token = eval_domains[i]
                emb_dict = dict(zip(seqs, embed_protein_vec_dataset(base_model, seqs, aspect_token)))
            elif model_type.lower() == 'double':
                emb_dict = dict(zip(seqs, embed_double_dataset(args, base_model, tokenizer, seqs)))
            else:
                emb_dict = dict(zip(seqs, embed_standard_plm(args, base_model, tokenizer, seqs)))

            base_model.to('cpu')  # Move base_model off the device
            torch.cuda.empty_cache()  # Clear the VRAM

            train_dataset = FineTuneDatasetEmbeds(args, emb_dict, train_sets[i][0], train_sets[i][1], task_types[i])
            valid_dataset = FineTuneDatasetEmbeds(args, emb_dict, valid_sets[i][0], valid_sets[i][1], task_types[i])
            test_dataset = FineTuneDatasetEmbeds(args, emb_dict, test_sets[i][0], test_sets[i][1], task_types[i])
            dataset = (train_dataset, valid_dataset, test_dataset)
            train_downstream_model(args, training_args, general_args, task_types, num_labels, dataset, i)


def evaluate_protein_vec(yargs, token):
    from models.protein_vec.src_run.huggingface_protein_vec import ProteinVec, ProteinVecConfig
    from transformers import T5Tokenizer
    
    tokenizer = T5Tokenizer.from_pretrained(yargs['general_args']['weight_path'])
    model = ProteinVec.from_pretrained(yargs['general_args']['weight_path'], config=ProteinVecConfig())

    model.to_eval()
    print(model)

    evaluate_contrastive_model(yargs,
                               model,
                               tokenizer,
                               compute_metrics_triplet,
                               get_datasets_test_triplet,
                               standard_data_collator,
                               token)
    evaluate_model_downstream(yargs, eval_config, model, tokenizer, token)
