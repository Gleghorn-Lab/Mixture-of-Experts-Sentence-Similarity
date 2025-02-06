import torch
import copy
import pickle
import os
import argparse
from torch.utils.data import ConcatDataset
from torchinfo import summary
from transformers import Trainer, TrainingArguments
from huggingface_hub import login


from data.get_data import get_all_train_data
from models.utils import prepare_model


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

DATA_DICT = {
    '[COPD]': 'GleghornLab/abstract_domain_copd',
    '[CVD]': 'GleghornLab/abstract_domain_cvd',
    '[CANCER]': 'GleghornLab/abstract_domain_skincancer',
    '[PARASITIC]': 'GleghornLab/abstract_domain_parasitic',
    '[AUTOIMMUNE]': 'GleghornLab/abstract_domain_autoimmune',
}

path_token_dict = {
    'GleghornLab/abstract_domain_copd': '[COPD]',
    'GleghornLab/abstract_domain_cvd': '[CVD]',
    'GleghornLab/abstract_domain_skincancer': '[CANCER]',
    'GleghornLab/abstract_domain_parasitic': '[PARASITIC]',
    'GleghornLab/abstract_domain_autoimmune': '[AUTOIMMUNE]'
}

token_expert_dict = {
    '[COPD]': 0,
    '[CVD]': 1,
    '[CANCER]': 2,
    '[PARASITIC]': 3,
    '[AUTOIMMUNE]': 4
}


def parse_args():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--model_path", type=str, default="Synthyra/ESMplusplus_small", help="Path to any base weights")
    parser.add_argument("--save_path", type=str, default="Synthyra/test", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--wandb_project", type=str, default="MOE_sentence_similarity", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of sequences fed to the model")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    args = parser.parse_args()
    return args


def main(args):
    model_path = 'answerdotai/ModernBERT-base'
    domains = list(DATA_DICT.keys())
    lora = True
    moe = True
    CV = 5
    add_tokens = True
    model, tokenizer = prepare_model(model_path, domains, lora, moe)
    summary(model)

    datasets = get_all_train_data(
        data_paths=list(DATA_DICT.values()),
        tokenizer=tokenizer,
        path_token_dict=path_token_dict,
        token_expert_dict=token_expert_dict,
        max_length=args.max_length,
        add_tokens=add_tokens,
        cross_validation=True,
        cv=CV,
    )

    for i, dataset in enumerate(datasets):
        eval_dataset = datasets[i]
        train_datasets = [dataset for j, dataset in enumerate(datasets) if j != i]
        train_dataset = ConcatDataset(train_datasets)

        ### Define Training Arguments
        training_args = TrainingArguments(
            output_dir=args.save_path.split('/')[-1],
            overwrite_output_dir=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=1,
            logging_steps=100,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=args.save_every,
            eval_steps=args.save_every // 2,
            logging_dir="./logs",
            learning_rate=args.lr,
            fp16=args.fp16,
            dataloader_num_workers=4 if not args.bugfix else 0,
            report_to="wandb" if WANDB_AVAILABLE else 'none',
            save_only_model=True,
            save_total_limit=3,
        )

        ### Create a trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics, 
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
        )

    ### Train
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print('Initial Metrics: \n', metrics)
    trainer.train()
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print('Final Metrics: \n', metrics)
    trainer.push_to_hub()
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    if WANDB_AVAILABLE:
        run_name = args.save_path.split('/')[-1]
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.token is not None:
        login(args.token)    

    if args.bugfix:
        args.batch_size = 4
        args.max_length = 32
        args.save_every = 100

    main(args)
