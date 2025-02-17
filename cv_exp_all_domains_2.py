import argparse
import os
import torch
import numpy as np
from torch.utils.data import ConcatDataset, Subset
from torchinfo import summary
from transformers import Trainer, TrainingArguments
from huggingface_hub import login
import pandas as pd

from data.data_collators import get_data_collator
from data.get_data import get_all_train_data, get_all_test_data
from models.utils import prepare_model
from metrics import compute_metrics_sentence_similarity_positives as compute_metrics

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Example dictionaries from your script
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
    parser.add_argument("--save_path", type=str, default="lhallee/moe_sim_test", help="Base path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--wandb_project", type=str, default="MOE_sentence_similarity", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_every", type=int, default=5000, help="Save the model every n steps and evaluate every n steps")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    args = parser.parse_args()
    return args


def main(args):
    # Experiment and CV settings
    model_path = 'answerdotai/ModernBERT-base'
    domains = list(DATA_DICT.keys())
    lora, CV = False, 3

    # Define your experiments: each tuple is (moe_setting, add_tokens_setting, clip_loss)
    moe_settings = [True, True]
    add_tokens_settings = [False, True]
    clip_losses = [True, False]
    experiments = list(zip(moe_settings, add_tokens_settings, clip_losses))
    
    # To collect all final evaluation metrics across runs
    all_run_metrics = []  # Each element will be a dict with keys: 'moe', 'add_tokens', 'clip_loss', 'cv_split', 'loss', 'sim_ratio', 'f1'
    
    # For each experiment (combination of MOE, add_tokens, and clip_loss settings)
    for moe_setting, add_tokens_setting, clip_loss in experiments:
        print(f"\n--- Running experiment: MOE = {moe_setting}, add_tokens = {add_tokens_setting}, clip_loss = {clip_loss} ---")
        
        # Get cross-validation splits (each a torch Dataset)
        cv_datasets = get_all_train_data(
            data_paths=list(DATA_DICT.values()),
            path_token_dict=path_token_dict,
            token_expert_dict=token_expert_dict,
            cross_validation=True,
            cv=CV,
        )

        test_dataset = get_all_test_data(
            data_paths=list(DATA_DICT.values()),
            path_token_dict=path_token_dict,
            token_expert_dict=token_expert_dict,
        )
    
        # Loop over each CV split: use one fold as eval and the rest as training
        for cv_index in range(len(cv_datasets)):
            print(f"\n--- CV split {cv_index+1}/{CV} ---")
            
            # Reinitialize a fresh model (and tokenizer) for each CV run, including the clip_loss setting
            model, tokenizer = prepare_model(model_path, domains, lora, moe_setting, clip_loss)
            summary(model)  # Print model summary

            data_collator = get_data_collator(tokenizer, domain_tokens=domains, max_length=args.max_length, add_tokens=add_tokens_setting)

            # Set up train and eval datasets
            full_eval_dataset = cv_datasets[cv_index]
            eval_dataset = Subset(full_eval_dataset, range(10000))
            full_eval_dataset = Subset(full_eval_dataset, range(100000))
            train_datasets = [cv_datasets[j] for j in range(len(cv_datasets)) if j != cv_index]
            train_dataset = ConcatDataset(train_datasets)

            # Create a unique run name and output directory (for wandb and hub)
            run_name = f"moe_{moe_setting}_addTokens_{add_tokens_setting}_clipLoss_{clip_loss}_cv_{cv_index}"
            unique_output_dir = os.path.join(args.save_path, run_name)
            os.makedirs(unique_output_dir, exist_ok=True)
            
            # Initialize a new wandb run if available
            if WANDB_AVAILABLE:
                wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            
            # --- Define TrainingArguments ---
            training_args = TrainingArguments(
                output_dir=unique_output_dir,
                overwrite_output_dir=True,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=1,
                logging_steps=100,
                save_strategy="steps",
                eval_strategy="steps",
                save_steps=args.save_every,
                eval_steps=args.save_every,
                logging_dir=os.path.join(unique_output_dir, "logs"),
                learning_rate=args.lr,
                fp16=args.fp16,
                dataloader_num_workers=4 if not args.bugfix else 0,
                report_to="wandb" if WANDB_AVAILABLE else 'none',
                save_only_model=True,
                save_total_limit=3,
            )
            
            # --- Create the Trainer ---
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            # (Optional) Evaluate before training to get initial metrics
            init_cv_metrics = trainer.evaluate(eval_dataset=eval_dataset)
            print("Initial CV Metrics:\n", init_cv_metrics)
            init_test_metrics = trainer.evaluate(eval_dataset=test_dataset)
            print("Initial Test Metrics:\n", init_test_metrics)

            # --- Train for one epoch ---
            trainer.train()
            
            # Evaluate after training
            final_cv_metrics = trainer.evaluate(eval_dataset=full_eval_dataset)
            print("Final CV Metrics:\n", final_cv_metrics)
            final_test_metrics = trainer.evaluate(eval_dataset=test_dataset)
            print("Final Test Metrics:\n", final_test_metrics)
            
            # Save metrics from the test dataset (tracking loss, sim_ratio, and F1 score)
            run_metrics = {
                'moe': moe_setting,
                'add_tokens': add_tokens_setting,
                'clip_loss': clip_loss,
                'cv_split': cv_index,
                'cv_loss': final_cv_metrics.get('eval_loss', None),
                'cv_sim_ratio': final_cv_metrics.get('eval_sim_ratio', None),
                'cv_f1': final_cv_metrics.get('eval_f1', None),
                'test_loss': final_test_metrics.get('eval_loss', None),
                'test_sim_ratio': final_test_metrics.get('eval_sim_ratio', None),
                'test_f1': final_test_metrics.get('eval_f1', None)

            }
            all_run_metrics.append(run_metrics)
            
            # Push model to Hugging Face Hub (each run gets its own repo thanks to unique output_dir)
            #trainer.push_to_hub()
            
            if WANDB_AVAILABLE:
                wandb.finish()
            
            # Free GPU memory
            trainer.accelerator.free_memory()
            torch.cuda.empty_cache()
    
    # --- After all experiments & CV splits: Aggregate and report final results ---
    # Group metrics by experiment setting (moe, add_tokens, clip_loss)
    grouped_results = defaultdict(list)
    for entry in all_run_metrics:
        key = (entry['moe'], entry['add_tokens'], entry['clip_loss'])
        grouped_results[key].append(entry)
    
    # Prepare table rows for each experiment setting
    table_rows = []
    for (moe_setting, add_tokens_setting, clip_loss), entries in grouped_results.items():
        cv_loss_values = [e['cv_loss'] for e in entries if e['cv_loss'] is not None]
        cv_sim_values = [e['cv_sim_ratio'] for e in entries if e['cv_sim_ratio'] is not None]
        cv_f1_values = [e['cv_f1'] for e in entries if e['cv_f1'] is not None]
        test_loss_values = [e['test_loss'] for e in entries if e['test_loss'] is not None]
        test_sim_values = [e['test_sim_ratio'] for e in entries if e['test_sim_ratio'] is not None]
        test_f1_values = [e['test_f1'] for e in entries if e['test_f1'] is not None]

        cv_loss_mean = np.mean(cv_loss_values) if cv_loss_values else float('nan')
        cv_loss_std = np.std(cv_loss_values) if cv_loss_values else float('nan')
        cv_sim_mean = np.mean(cv_sim_values) if cv_sim_values else float('nan')
        cv_sim_std = np.std(cv_sim_values) if cv_sim_values else float('nan')
        cv_f1_mean = np.mean(cv_f1_values) if cv_f1_values else float('nan')
        cv_f1_std = np.std(cv_f1_values) if cv_f1_values else float('nan')
        test_loss_mean = np.mean(test_loss_values) if test_loss_values else float('nan')
        test_loss_std = np.std(test_loss_values) if test_loss_values else float('nan')
        test_sim_mean = np.mean(test_sim_values) if test_sim_values else float('nan')
        test_sim_std = np.std(test_sim_values) if test_sim_values else float('nan')
        test_f1_mean = np.mean(test_f1_values) if test_f1_values else float('nan')
        test_f1_std = np.std(test_f1_values) if test_f1_values else float('nan')
        

        table_rows.append({
            'MOE': str(moe_setting),
            'Add_Tokens': str(add_tokens_setting),
            'Clip_Loss': str(clip_loss),
            'CV_Loss_mean': f"{cv_loss_mean:.4f}",
            'CV_Loss_std': f"{cv_loss_std:.4f}",
            'CV_Sim_Ratio_mean': f"{cv_sim_mean:.4f}",
            'CV_Sim_Ratio_std': f"{cv_sim_std:.4f}",
            'CV_F1_mean': f"{cv_f1_mean:.4f}",
            'CV_F1_std': f"{cv_f1_std:.4f}",
            'Test_Loss_mean': f"{test_loss_mean:.4f}",
            'Test_Loss_std': f"{test_loss_std:.4f}",
            'Test_Sim_Ratio_mean': f"{test_sim_mean:.4f}",
            'Test_Sim_Ratio_std': f"{test_sim_std:.4f}",
            'Test_F1_mean': f"{test_f1_mean:.4f}",
            'Test_F1_std': f"{test_f1_std:.4f}"
        })
    
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(table_rows)
    results_file = os.path.join(args.save_path, "final_results.csv")
    os.makedirs(args.save_path, exist_ok=True)
    df.to_csv(results_file, index=False)
    print(f"\nFinal results saved to {results_file}")


if __name__ == "__main__":
    args = parse_args()

    if args.token is not None:
        login(args.token)

    if args.bugfix:
        args.batch_size = 4
        args.max_length = 32
        args.save_every = 100

    main(args)
