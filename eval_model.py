import argparse
import os
import torch
import torch.nn.functional as F
from torchinfo import summary
from transformers import Trainer, TrainingArguments
from huggingface_hub import login
from data.data_collators import get_data_collator
from data.get_data import get_single_eval_data
from models.utils import load_from_pretrained
from metrics import compute_metrics_sentence_similarity_with_negatives as compute_metrics
from metrics import compute_metrics_benchmark

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


DATA_DICT = {
    '[COPD]': 'GleghornLab/abstract_domain_copd',
    '[CVD]': 'GleghornLab/abstract_domain_cvd',
    '[CANCER]': 'GleghornLab/abstract_domain_skincancer',
    '[PARASITIC]': 'GleghornLab/abstract_domain_parasitic',
    '[AUTOIMMUNE]': 'GleghornLab/abstract_domain_autoimmune',
}

PATH_TOKEN_DICT = {
    'GleghornLab/abstract_domain_copd': '[COPD]',
    'GleghornLab/abstract_domain_cvd': '[CVD]',
    'GleghornLab/abstract_domain_skincancer': '[CANCER]',
    'GleghornLab/abstract_domain_parasitic': '[PARASITIC]',
    'GleghornLab/abstract_domain_autoimmune': '[AUTOIMMUNE]'
}

TOKEN_EXPERT_DICT = {
    '[COPD]': 0,
    '[CVD]': 1,
    '[CANCER]': 2,
    '[PARASITIC]': 3,
    '[AUTOIMMUNE]': 4
}

MODELS_TO_EVALUATE = [
    'lhallee/se_train_run_COPD',
    'lhallee/moe_train_run',
    'lhallee/se_train_run_PARASITIC',
    'lhallee/se_train_run_AUTOIMMUNE',
    'lhallee/se_train_run_CVD',
    'lhallee/se_train_run_CANCER',
]


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
    parser.add_argument("--loss_type", type=str, default='mnr_plus', help="Loss type")
    args = parser.parse_args()
    return args


def main(args):
    base_model_path = 'answerdotai/ModernBERT-base'
    domains = list(DATA_DICT.keys())

    for model_path in MODELS_TO_EVALUATE:
        if 'moe' in model_path:
            lora, moe, add_tokens = False, True, True
        else:
            lora, moe, add_tokens = False, False, True

        for domain, data_path in DATA_DICT.items():
            if 'moe' not in model_path.lower():
                # We use the models special token for all datasets if not MOE
                token = '[' + model_path.split('_')[-1] + ']'
                path_token_dict = {
                    'GleghornLab/abstract_domain_copd': token,
                    'GleghornLab/abstract_domain_cvd': token,
                    'GleghornLab/abstract_domain_skincancer': token,
                    'GleghornLab/abstract_domain_parasitic': token,
                    'GleghornLab/abstract_domain_autoimmune': token,
                }
            else:
                path_token_dict = PATH_TOKEN_DICT

            eval_dataset = get_single_eval_data(
                data_path=data_path,
                path_token_dict=path_token_dict,
                token_expert_dict=TOKEN_EXPERT_DICT,
            )            
            model, tokenizer = load_from_pretrained(base_model_path, model_path, domains, lora, moe, args.loss_type)
            data_collator = get_data_collator(tokenizer, domain_tokens=domains, max_length=args.max_length, add_tokens=add_tokens)

            domain = domain.replace('[', '').replace(']', '')
            model_name = model_path.split('/')[-1]
            run_name = f"{model_name}_{domain}"
            print(run_name)
            unique_output_dir = os.path.join(args.save_path, run_name)
            os.makedirs(unique_output_dir, exist_ok=True)
            
            training_args = TrainingArguments(
                output_dir=unique_output_dir,
                overwrite_output_dir=True,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=1,
                logging_steps=100,
                save_strategy="steps",
                save_steps=args.save_every,
                logging_dir=os.path.join(unique_output_dir, "logs"),
                learning_rate=args.lr,
                dataloader_num_workers=4 if not args.bugfix else 0,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            preds, labels, metrics = trainer.predict(test_dataset=eval_dataset)
            print(metrics)

            break
            trainer.accelerator.free_memory()
            torch.cuda.empty_cache()

    
if __name__ == "__main__":
    args = parse_args()

    if args.token is not None:
        login(args.token)

    if args.bugfix:
        args.batch_size = 4
        args.max_length = 32
        args.save_every = 100

    main(args)
