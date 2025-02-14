import argparse
import os
import torch
from torchinfo import summary
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from huggingface_hub import login

from data.data_collators import get_data_collator
from data.get_data import get_all_eval_data, get_all_train_data, get_single_train_data, get_single_eval_data
from models.utils import prepare_model
from models.modeling_moe_bert import MoEBertForSentenceSimilarity
from metrics import compute_metrics_sentence_similarity_with_negatives as compute_metrics

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
    parser = argparse.ArgumentParser(description="Full Experiment Runner")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--base_path", type=str, default="GleghornLab", help="Base path for saving models")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--wandb_project", type=str, default="MOE_PUBLISH", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_every", type=int, default=5000, help="Save/eval steps for joint training")
    parser.add_argument("--save_every_domain", type=int, default=2500, help="Save/eval steps for domain training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument("--loss_type", type=str, default='mnr_plus', help="Loss type")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--skip_domain", action="store_true", help="Skip domain-specific training")
    parser.add_argument("--skip_joint", action="store_true", help="Skip joint training")
    args = parser.parse_args()
    return args


def train_and_evaluate(trainer, eval_dataset, model_name, domain=None):
    domain_str = f" for {domain}" if domain else ""
    print(f"\nStarting training for {model_name}{domain_str}")
    init_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Initial Metrics{domain_str}:\n", init_metrics)
    
    trainer.train()
    
    final_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Final Metrics{domain_str}:\n", final_metrics)
    return final_metrics


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, args, run_name, save_path, domains, add_tokens, save_steps):
    data_collator = get_data_collator(tokenizer, domain_tokens=domains, max_length=args.max_length, add_tokens=add_tokens)
    
    unique_output_dir = os.path.join(save_path, run_name)
    os.makedirs(unique_output_dir, exist_ok=True)
    
    if WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    training_args = TrainingArguments(
        output_dir=unique_output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=1,
        logging_steps=100,
        save_strategy="steps",
        eval_strategy="steps", 
        save_steps=save_steps,
        eval_steps=save_steps,
        logging_dir=os.path.join(unique_output_dir, "logs"),
        learning_rate=args.lr,
        fp16=args.fp16,
        dataloader_num_workers=4 if not args.bugfix else 0,
        report_to="wandb" if WANDB_AVAILABLE else 'none',
        save_only_model=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )


def save_model(trainer, tokenizer, save_path, eval_dataset):
    trainer.model.push_to_hub(save_path)
    tokenizer.push_to_hub(save_path)
    
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()

    # Load and evaluate saved model
    model = MoEBertForSentenceSimilarity.from_pretrained(save_path).cuda().eval()
    trainer.model = model
    loaded_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Loaded Model Metrics:\n", loaded_metrics)


def run_domain_specific_training(args):
    print("\n=== Starting Domain-Specific Training ===")
    model_path = 'answerdotai/ModernBERT-base'
    domains = list(DATA_DICT.keys())
    
    for domain, data_path in DATA_DICT.items():
        train_dataset = get_single_train_data(
            data_path=data_path,
            path_token_dict=path_token_dict,
            token_expert_dict=token_expert_dict,
            cross_validation=False,
            cv=1,
        )

        eval_dataset = get_single_eval_data(
            data_path=data_path,
            path_token_dict=path_token_dict,
            token_expert_dict=token_expert_dict,
        )
        
        model, tokenizer = prepare_model(model_path, domains, lora=False, moe=False, loss_type=args.loss_type)
        summary(model)

        domain_clean = domain.replace('[', '').replace(']', '')
        run_name = f"se_domain_{domain_clean}"
        save_path = os.path.join(args.base_path, "se_domain")
        
        trainer = setup_trainer(
            model, tokenizer, train_dataset, eval_dataset, args, 
            run_name, save_path, domains, add_tokens=True, 
            save_steps=args.save_every_domain
        )
        
        _ = train_and_evaluate(trainer, eval_dataset, "SE Domain", domain_clean)
        save_path = f"{save_path}HF_{domain_clean}"
        save_model(trainer, tokenizer, f"{save_path}HF_{domain_clean}", eval_dataset)
        
        if WANDB_AVAILABLE:
            wandb.finish()
        trainer.accelerator.free_memory()
        torch.cuda.empty_cache()


def run_joint_training(args):
    print("\n=== Starting Joint Training ===")
    model_path = 'answerdotai/ModernBERT-base'
    domains = list(DATA_DICT.keys())
    
    train_dataset = get_all_train_data(
        data_paths=list(DATA_DICT.values()),
        path_token_dict=path_token_dict,
        token_expert_dict=token_expert_dict,
        cross_validation=False,
        cv=1,
    )

    eval_dataset = get_all_eval_data(
        data_paths=list(DATA_DICT.values()),
        path_token_dict=path_token_dict,
        token_expert_dict=token_expert_dict,
    )

    # SE training with tokens
    model, tokenizer = prepare_model(model_path, domains, lora=False, moe=False, loss_type=args.loss_type)
    trainer = setup_trainer(
        model, tokenizer, train_dataset, eval_dataset, args,
        "se_all_with_tokens", os.path.join(args.base_path, "se_all_tokens"),
        domains, add_tokens=True, save_steps=args.save_every
    )
    _ = train_and_evaluate(trainer, eval_dataset, "SE All (with tokens)")
    save_model(trainer, tokenizer, f"{args.base_path}/se_all_tokensHF", eval_dataset)
    if WANDB_AVAILABLE: wandb.finish()
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()

    # SE training without tokens
    model, tokenizer = prepare_model(model_path, domains, lora=False, moe=False, loss_type=args.loss_type)
    trainer = setup_trainer(
        model, tokenizer, train_dataset, eval_dataset, args,
        "se_all_no_tokens", os.path.join(args.base_path, "se_all_no_tokens"),
        domains, add_tokens=False, save_steps=args.save_every
    )
    _ = train_and_evaluate(trainer, eval_dataset, "SE All (no tokens)")
    save_model(trainer, tokenizer, f"{args.base_path}/se_all_no_tokensHF", eval_dataset)
    if WANDB_AVAILABLE: wandb.finish()
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()

    # MoE training with tokens
    model, tokenizer = prepare_model(model_path, domains, lora=False, moe=True, loss_type=args.loss_type)
    trainer = setup_trainer(
        model, tokenizer, train_dataset, eval_dataset, args,
        "moe_with_tokens", os.path.join(args.base_path, "moe_tokens"),
        domains, add_tokens=True, save_steps=args.save_every
    )
    _ = train_and_evaluate(trainer, eval_dataset, "MoE (with tokens)")
    save_model(trainer, tokenizer, f"{args.base_path}/moe_tokensHF", eval_dataset)
    if WANDB_AVAILABLE: wandb.finish()
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()

    # MoE training without tokens
    model, tokenizer = prepare_model(model_path, domains, lora=False, moe=True, loss_type=args.loss_type)
    trainer = setup_trainer(
        model, tokenizer, train_dataset, eval_dataset, args,
        "moe_no_tokens", os.path.join(args.base_path, "moe_no_tokens"),
        domains, add_tokens=False, save_steps=args.save_every
    )
    _ = train_and_evaluate(trainer, eval_dataset, "MoE (no tokens)")
    save_model(trainer, tokenizer, f"{args.base_path}/moe_no_tokensHF", eval_dataset)
    if WANDB_AVAILABLE: wandb.finish()
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()


def main(args):
    if not args.skip_domain:
        run_domain_specific_training(args)
    
    if not args.skip_joint:
        run_joint_training(args)


if __name__ == "__main__":
    args = parse_args()

    if args.token is not None:
        login(args.token)

    if args.bugfix:
        args.batch_size = 4
        args.max_length = 32
        args.save_every = 100
        args.save_every_domain = 100

    main(args)
