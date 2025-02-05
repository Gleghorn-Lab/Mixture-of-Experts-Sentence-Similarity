import os
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict

from metrics import compute_metrics_benchmark
from data.get_data import get_all_eval_documents

from models.embedding_models import (
    model_to_class_dict
)

DATA_DICT = {
    '[COPD]': 'GleghornLab/abstract_domain_copd',
    '[CVD]': 'GleghornLab/abstract_domain_cvd',
    '[CANCER]': 'GleghornLab/abstract_domain_skincancer',
    '[PARASITIC]': 'GleghornLab/abstract_domain_parasitic',
    '[AUTOIMMUNE]': 'GleghornLab/abstract_domain_autoimmune',
}

MODEL_DICT = {
    #'ModernBERT-base': 'answerdotai/ModernBERT-base',
    #'ModernBERT-large': 'answerdotai/ModernBERT-large',
    'BERT-base': 'google-bert/bert-base-uncased',
    #'BERT-large': 'google-bert/bert-large-uncased',
    'Mini': 'sentence-transformers/all-MiniLM-L6-v2',
    #'MPNet': 'sentence-transformers/all-mpnet-base-v2',
    #'RoBERTa-base': 'FacebookAI/roberta-base',
    #'RoBERTa-large': 'FacebookAI/roberta-large',
    #'DeBERTa-v3-base': 'microsoft/deberta-v3-base',
    #'Llama-3.2-1B': 'meta-llama/Llama-3.2-1B',
    #'SciBERT': 'allenai/scibert_scivocab_uncased',
    #'PubmedBERT': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    #'BioBERT': 'dmis-lab/biobert-v1.1',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(result_dir, test_mode=False):
    """
    Evaluate multiple embedding models on several datasets, compute cosine-similarity-based predictions,
    and output both per-example predictions and computed metrics to CSV files.

    For each model and each dataset (domain), two CSV files are produced:
      - predictions CSV: containing each prediction and its label,
      - metrics CSV: containing the aggregated metrics (as computed by compute_metrics_benchmark).

    In addition, an aggregated (i.e. "all datasets") predictions and metrics CSV are generated per model.
    Finally, a summary CSV (saved in the result_dir) is created containing a table of metrics across all models
    and domains for easy publication.

    Args:
        result_dir: Directory to save results
        test_mode: If True, uses random embeddings instead of actual model embeddings
    """
    # Load evaluation documents. Each list must be aligned such that the i-th example in each list corresponds.
    all_a_documents, all_b_documents, all_domain_tokens, all_labels = get_all_eval_documents(DATA_DICT)
    texts = set(all_a_documents + all_b_documents)

    # This list will collect summary metric records for all model/dataset combinations.
    summary_records = []

    # Loop over each model you want to evaluate.
    for model_name, model_path in MODEL_DICT.items():
        print(f"Processing model: {model_name}")

        # Create a directory for the current model's output.
        model_dir = os.path.join(result_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        if test_mode:
            # Generate random embeddings dictionary
            embeddings_dict = {text: torch.randn(1, 128, dtype=torch.float32) for text in texts}
        else:
            # Instantiate the embedder.
            model_class = model_to_class_dict[model_name]
            embedder = model_class(model_path).to(DEVICE)
            embeddings_dict = embedder.embed_dataset(
                texts,
                embedder.tokenizer,
                batch_size=2,
                max_len=1024,
                cls_pooling=False
            )

        # Prepare dictionaries to hold results per domain and overall.
        result_dict = defaultdict(lambda: {'preds': [], 'labels': []})
        aggregated_preds = []
        aggregated_labels = []
        aggregated_rows = []  # To record domain info with each prediction for later aggregation.

        # Iterate through each evaluation example.
        for a, b, domain_token, label in zip(all_a_documents, all_b_documents, all_domain_tokens, all_labels):
            a_emb = embeddings_dict[a]
            b_emb = embeddings_dict[b]
            # Compute cosine similarity. (Make sure the dimension is set appropriately.)
            cosine_sim = F.cosine_similarity(a_emb, b_emb, dim=1)
            sim_value = cosine_sim.item()

            # Append per-domain results.
            result_dict[domain_token]['preds'].append(sim_value)
            result_dict[domain_token]['labels'].append(label)

            # Append to the aggregated results.
            aggregated_preds.append(sim_value)
            aggregated_labels.append(label)
            domain_clean = domain_token.strip("[]")
            aggregated_rows.append({
                "domain": domain_clean,
                "prediction": sim_value,
                "label": label
            })

        # Process and save results for each individual domain.
        for domain_token, results in result_dict.items():
            domain_clean = domain_token.strip("[]")
            metrics = compute_metrics_benchmark(results['preds'], results['labels'])
            result_dict[domain_token]['metrics'] = metrics

            # Save per-example predictions.
            df_preds = pd.DataFrame({
                "prediction": results['preds'],
                "label": results['labels']
            })
            preds_file = os.path.join(model_dir, f"{model_name}_{domain_clean}_predictions.csv")
            df_preds.to_csv(preds_file, index=False)

            # Save the computed metrics.
            df_metrics = pd.DataFrame([metrics])
            metrics_file = os.path.join(model_dir, f"{model_name}_{domain_clean}_metrics.csv")
            df_metrics.to_csv(metrics_file, index=False)

            # Record summary for the current domain.
            summary_record = {"Model": model_name, "Dataset": domain_clean}
            summary_record.update(metrics)
            summary_records.append(summary_record)

        # Compute and save aggregated metrics (i.e. across all domains).
        aggregated_metrics = compute_metrics_benchmark(aggregated_preds, aggregated_labels)
        
        # Save aggregated per-example predictions (with domain info).
        df_agg_preds = pd.DataFrame(aggregated_rows)
        agg_preds_file = os.path.join(model_dir, f"{model_name}_all_predictions.csv")
        df_agg_preds.to_csv(agg_preds_file, index=False)
        
        # Save aggregated metrics.
        df_agg_metrics = pd.DataFrame([aggregated_metrics])
        agg_metrics_file = os.path.join(model_dir, f"{model_name}_all_metrics.csv")
        df_agg_metrics.to_csv(agg_metrics_file, index=False)

        # Record aggregated summary metrics.
        summary_record = {"Model": model_name, "Dataset": "All"}
        summary_record.update(aggregated_metrics)
        summary_records.append(summary_record)

    # Save the overall summary CSV.
    summary_df = pd.DataFrame(summary_records)
    summary_csv_file = os.path.join(result_dir, "summary.csv")
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Summary saved to {summary_csv_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate embedding models and output predictions and metrics to CSV files."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to save the CSV output files."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with random embeddings"
    )
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    main(args.result_dir, args.test)
