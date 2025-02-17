import argparse
import os
import torch
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, EvalPrediction
from huggingface_hub import login
from data.data_collators import get_data_collator
from data.get_data import get_single_test_data
from models.modeling_moe_bert import MoEBertForSentenceSimilarity
from metrics import compute_metrics_sentence_similarity_with_negatives as compute_metrics
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# These dictionaries are used for token/domain lookup.
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

# List of model paths to evaluate.
MODELS_TO_EVALUATE = [
    'GleghornLab/moe_tokensHF',
    'GleghornLab/se_all_tokensHF',
    'GleghornLab/se_domainHF_CVDHF_CVD',
    'GleghornLab/se_domainHF_AUTOIMMUNEHF_AUTOIMMUNE',
    'GleghornLab/se_domainHF_CANCERHF_CANCER',
    'GleghornLab/se_domainHF_PARASITICHF_PARASITIC',
    'GleghornLab/se_domainHF_COPDHF_COPD',
]


def parse_args():
    parser = argparse.ArgumentParser(description="MOE Sentence Similarity Evaluator")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    args = parser.parse_args()
    return args


def main(args):
    # This list will collect summary metric records for all model/domain combinations.
    summary_records = []

    # Loop over each model.
    for model_path in MODELS_TO_EVALUATE:
        # Load model and tokenizer.
        model = MoEBertForSentenceSimilarity.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_name = model_path.split('/')[-1]

        # Create a directory to hold all CSV outputs for the current model.
        model_dir = os.path.join("results", "trained_models", model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Initialize lists to store domain metrics and sample counts
        domain_metrics_list = []
        domain_sample_counts = []
        aggregated_rows = []  # each row will contain domain, prediction, and label

        # Evaluate on each domain.
        for domain, data_path in DATA_DICT.items():
            # If the model is not MOE, override tokens using the model name.
            if 'moe' not in model_path.lower() and 'all' not in model_path.lower():
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

            # Load evaluation dataset for the current domain.
            eval_dataset = get_single_test_data(
                data_path=data_path,
                path_token_dict=path_token_dict,
                token_expert_dict=TOKEN_EXPERT_DICT,
            )
            data_collator = get_data_collator(tokenizer, 
                                              domain_tokens=list(DATA_DICT.keys()), 
                                              max_length=args.max_length, 
                                              add_tokens=True)
            domain_clean = domain.strip("[]")
            run_name = f"{model_name}_{domain_clean}"
            print(f"Evaluating {run_name}")
            
            training_args = TrainingArguments(
                output_dir='holder',
                overwrite_output_dir=True,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=1,
                learning_rate=1e-4,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            # Run prediction on the evaluation set.
            embeddings, labels, domain_metrics = trainer.predict(test_dataset=eval_dataset)
            # Split the returned embeddings into the two sentence parts.
            emb_a, emb_b = torch.chunk(torch.tensor(embeddings), 2, dim=1)
            preds = torch.cosine_similarity(emb_a, emb_b, dim=-1)
            preds = preds.tolist()
            labels = labels.tolist()
            
            # Save per-example predictions (and true labels) to CSV.
            df_preds = pd.DataFrame({
                "prediction": preds,
                "label": labels
            })
            preds_file = os.path.join(model_dir, f"{model_name}_{domain_clean}_predictions.csv")
            df_preds.to_csv(preds_file, index=False)
            df_metrics = pd.DataFrame([domain_metrics])
            metrics_file = os.path.join(model_dir, f"{model_name}_{domain_clean}_metrics.csv")
            df_metrics.to_csv(metrics_file, index=False)
            
            # Store predictions with domain info
            for p, l in zip(preds, labels):
                aggregated_rows.append({
                    "domain": domain_clean,
                    "prediction": p,
                    "label": l
                })
            
            # Record a summary for this domain.
            summary_record = {"Model": model_name, "Dataset": domain_clean}
            summary_record.update(domain_metrics)
            summary_records.append(summary_record)
            
            # Store domain metrics and sample count for weighted average
            domain_metrics_list.append(domain_metrics)
            domain_sample_counts.append(len(labels))
            
            trainer.accelerator.free_memory()
            torch.cuda.empty_cache()
        
        # Calculate weighted average of metrics across domains
        total_samples = sum(domain_sample_counts)
        aggregated_metrics = {}
        for metric in domain_metrics_list[0].keys():
            weighted_sum = sum(metrics[metric] * count for metrics, count 
                             in zip(domain_metrics_list, domain_sample_counts))
            aggregated_metrics[metric] = weighted_sum / total_samples
        
        # Save aggregated predictions
        df_agg_preds = pd.DataFrame(aggregated_rows)
        agg_preds_file = os.path.join(model_dir, f"{model_name}_all_predictions.csv")
        df_agg_preds.to_csv(agg_preds_file, index=False)
        
        # Save aggregated metrics
        df_agg_metrics = pd.DataFrame([aggregated_metrics])
        agg_metrics_file = os.path.join(model_dir, f"{model_name}_all_metrics.csv")
        df_agg_metrics.to_csv(agg_metrics_file, index=False)
        
        summary_record = {"Model": model_name, "Dataset": "All"}
        summary_record.update(aggregated_metrics)
        summary_records.append(summary_record)
    
    # Save overall summary CSV (aggregated across models and domains).
    summary_csv_file = os.path.join("results", "trained_models", "summary.csv")
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(summary_csv_file, index=False)
    print(f"Summary saved to {summary_csv_file}")


if __name__ == "__main__":
    args = parse_args()

    if args.token is not None:
        login(args.token)

    if args.bugfix:
        args.batch_size = 4
        args.max_length = 32
        args.save_every = 100

    main(args)
