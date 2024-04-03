import os
import argparse
import inspect
from pathlib import Path
from dataclasses import fields, dataclass, field
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from data_protein_vec import collate_fn, get_parquet, construct_datasets
from model_protein_moe import trans_basic_block, trans_basic_block_Config
from utils import SessionTree

#Comand line
def arguments():
    parser = argparse.ArgumentParser(description="Train a structural embedding model")
    
    parser.add_argument("--gpus",
            type=int,
            help="Num. gpus",
            default=1
    )
    parser.add_argument("--nodes",
            type=int,
            help="Num. nodes",
            default=1
    )

    parser.add_argument("--data",
            type=Path,
            required=True,
            help="Data"
    )


    parser.add_argument("--embeddings",
            type=Path,
            required=True,
            help="Embeddings path"
    )
    
    parser.add_argument("--session",
            type=Path,
            required=True,
            help="Training session directory; models are saved here along with other important metadata"
    )

    parser.add_argument("--batch-size",
            type=int,
            help="Batch size",
            default=1
    )

    parser.add_argument("--max-epochs",
            type=int,
            help="Epochs",
            default=5
    )


    parser.add_argument("--seed",
            type=int,
            help="Random seed",
            default=1230
    )
    parser.add_argument("--train-prop",
            type=float,
            default=0.9,
            help="Proportion of dataset used to train"
    )
    parser.add_argument("--val-prop",
            type=float,
            default=0.05,
            help="Proportion of data to use for validation"
    )

    parser.add_argument("--test-prop",
            type=float,
            default=0.05,
            help="Proportion of data to use for test"
    )

    # Now add the transformer model arguments
    for field in fields(trans_basic_block_Config):
        parser.add_argument(
            f"--{field.name}", default=field.default, type=field.type
        )

    return parser.parse_args()


def collect_trans_block_arguments(args) -> trans_basic_block_Config:
        trans_block_conf_args = inspect.signature(trans_basic_block_Config).parameters
        return {k: v for k, v in args.items() if k in trans_block_conf_args}

if __name__ == '__main__':
        
        #Construct datasets: Make train, test, and validation datasets
        args = arguments()
        config = collect_trans_block_arguments(vars(args))
        config = trans_basic_block_Config(**config)
        
        print(config, flush=True)
        model = config.build()
        tree = SessionTree(args.session)
        config.to_json(tree.params)

        train_ds, val_ds, test_ds = construct_datasets(args.data,  args.embeddings, args.train_prop, args.val_prop, args.test_prop)
        print("Constructed datasets")
        #Build the data loaders: train data loader and validation data loader
        train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=False, persistent_workers=False)
        val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=False, persistent_workers=False)

        val_check_interval = 0.02
        accumulate_grad_batches = 8
        effective_batch_size = args.gpus * args.nodes * args.batch_size * accumulate_grad_batches
        every_n_train_steps = int(len(train_ds) * val_check_interval / effective_batch_size)
        print("Saving and validating every ", every_n_train_steps, " steps")

        #Model checkpoints
        ckpt = pl.callbacks.ModelCheckpoint(
                dirpath=tree.checkpoints,
                monitor="val_loss",
                verbose=True,
                filename="{epoch}-{step}-{val_loss:0.4f}",
                every_n_train_steps=every_n_train_steps,
                save_top_k=5,
                save_weights_only=False,
                save_last=True
        )

        #Logger
        logger = pl.loggers.TensorBoardLogger(tree.logs)

        checkpoint_path = str(tree.checkpoints) + '/last.ckpt'
        #Trainer
        trainer = pl.Trainer(
                accelerator="cuda",
                strategy=DDPStrategy(find_unused_parameters=False),
                precision='32',
                num_sanity_val_steps=0,
                callbacks=[ckpt],
                logger=logger,
                gpus=args.gpus,
                val_check_interval=val_check_interval,
                num_nodes=args.nodes,
                gradient_clip_val=0.5,
                gradient_clip_algorithm="norm",
                max_epochs=args.max_epochs,
                accumulate_grad_batches=accumulate_grad_batches
        )
        #Setup model and fit 
        print("Training...")
        trainer.fit(model, train_dataloader, val_dataloader)
