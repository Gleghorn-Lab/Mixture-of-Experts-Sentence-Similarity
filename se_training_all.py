import argparse
import os
import torch
from torchinfo import summary
from transformers import Trainer, TrainingArguments
from huggingface_hub import login

from data.data_collators import get_data_collator
from data.get_data import get_all_test_data, get_all_train_data
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
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--save_path", type=str, default="GleghornLab/se_all", help="Base path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--wandb_project", type=str, default="MOE_FINAL", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_every", type=int, default=10000, help="Save the model every n steps and evaluate every n steps")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument("--loss_type", type=str, default='mnr_plus', help="Loss type")
    args = parser.parse_args()
    return args


def main(args):
    model_path = 'answerdotai/ModernBERT-base'
    domains = list(DATA_DICT.keys())
    lora, moe, add_tokens = False, False, True

    train_dataset = get_all_train_data(
        data_paths=list(DATA_DICT.values()),
        path_token_dict=path_token_dict,
        token_expert_dict=token_expert_dict,
        cross_validation=False,
        cv=1,
    )

    eval_dataset = get_all_test_data(
        data_paths=list(DATA_DICT.values()),
        path_token_dict=path_token_dict,
        token_expert_dict=token_expert_dict,
    )
    
    model, tokenizer = prepare_model(model_path, domains, lora, moe, args.loss_type)
    summary(model)

    data_collator = get_data_collator(tokenizer, domain_tokens=domains, max_length=args.max_length, add_tokens=add_tokens)

    run_name = f"se_train_run_all"
    unique_output_dir = os.path.join(args.save_path, run_name)
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
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    init_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Initial Metrics:\n", init_metrics)
    
    trainer.train()
    
    final_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Final Metrics:\n", final_metrics)
    
    save_path = args.save_path + f'HF'
    trainer.model.push_to_hub(save_path)
    tokenizer.push_to_hub(save_path)
    
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()

    # Add model loading and evaluation
    model = MoEBertForSentenceSimilarity.from_pretrained(save_path).cuda().eval()
    trainer.model = model

    loaded_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Loaded Metrics:\n", loaded_metrics)
    
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    if args.token is not None:
        login(args.token)

    if args.bugfix:
        args.batch_size = 4
        args.max_length = 32
        args.save_every = 100

    main(args)
