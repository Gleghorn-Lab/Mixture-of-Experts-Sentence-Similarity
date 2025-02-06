#--------------------------
# TRAINER TEMPLATE
#--------------------------
import torch
import copy
import pickle
import os
import argparse
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from huggingface_hub import login
from datasets import load_dataset
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False





def parse_args():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--model_path", type=str, default="Synthyra/ESMplusplus_small", help="Path to any base weights")
    parser.add_argument("--save_path", type=str, default="Synthyra/test", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--wandb_project", type=str, default="test", help="Wandb project name")
    parser.add_argument("--dataset_name", type=str, default="Synthyra/omg_prot50", help="Name of the dataset to use for training")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of sequences fed to the model")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Load Model


    ### Load Dataset

    ### Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
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
        push_to_hub=False if args.bugfix else True,
        hub_always_push=False if args.bugfix else True,
        save_only_model=True,
        hub_strategy='every_save',
        hub_model_id=args.save_path,
        hub_private_repo=True,
        save_total_limit=3,
        # load_best_model_at_end=True,
        # metric_for_best_model='eval_loss',
        # greater_is_better=False,
    )

    ### Create a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics, 
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    ### Train
    metrics = trainer.evaluate(test_dataset)
    print('Initial Metrics: \n', metrics)
    trainer.train()
    metrics = trainer.evaluate(test_dataset)
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
        args.batch_size = 2
        args.max_length = 256
        args.save_every = 1000

    main(args)
