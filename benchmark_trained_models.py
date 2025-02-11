import os
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from metrics import compute_metrics_benchmark
from data.get_data import get_all_eval_documents, token_expert_dict

from models.embedding_models import MoeModernBertEmbedder


DATA_DICT = {
    '[COPD]': 'GleghornLab/abstract_domain_copd',
    '[CVD]': 'GleghornLab/abstract_domain_cvd',
    '[CANCER]': 'GleghornLab/abstract_domain_skincancer',
    '[PARASITIC]': 'GleghornLab/abstract_domain_parasitic',
    '[AUTOIMMUNE]': 'GleghornLab/abstract_domain_autoimmune',
}

MODEL_DICT = {
    'MOE': 'lhallee/moe_train_run',
    'SE-Autoimmune': 'lhallee/se_train_run_AUTOIMMUNE',
    'SE-Cancer': 'lhallee/se_train_run_CANCER',
    'SE-CVD': 'lhallee/se_train_run_CVD',
    'SE-Parasitic': 'lhallee/se_train_run_PARASITIC',
    'SE-COPD': 'lhallee/se_train_run_COPD',
}

MODEL_DOMAIN_DICT = {
    'SE-Autoimmune': '[AUTOIMMUNE]',
    'SE-Cancer': '[CANCER]',
    'SE-CVD': '[CVD]',
    'SE-Parasitic': '[PARASITIC]',
    'SE-COPD': '[COPD]',
}


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
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
        max_length: Maximum length of text to embed
    """
    result_dir = args.result_dir
    test_mode = args.test
    max_length = args.max_length

    # Load evaluation documents. Each list must be aligned such that the i-th example in each list corresponds.
    # Retrieve documents, domain tokens, labels, and expert assignments.
    all_a_documents, all_b_documents, all_domain_tokens, all_labels, all_expert_assignments = get_all_eval_documents(DATA_DICT, token_expert_dict)

    # Trim the documents.
    all_a_documents = [doc[:max_length].strip() for doc in all_a_documents]
    all_b_documents = [doc[:max_length].strip() for doc in all_b_documents]

    # a and b documents of the same index share a domain token and expert assignment.
    # So, we combine them by duplicating the domain tokens and expert assignments.
    docs_combined = all_a_documents + all_b_documents
    domain_tokens_combined = all_domain_tokens + all_domain_tokens
    expert_assignments_combined = all_expert_assignments + all_expert_assignments

    # (Optional) Verify that the lengths match.
    assert len(docs_combined) == len(domain_tokens_combined) == len(expert_assignments_combined), \
        "Length mismatch between docs, domain tokens, or expert assignments!"

    # Prepend the domain token to each document and append [SEP] at the end.
    modified_docs = [
        f"{domain_token}{doc}[SEP]"
        for domain_token, doc in zip(domain_tokens_combined, docs_combined)
    ]

    modified_a_docs = [
        f"{domain_token}{doc}[SEP]"
        for domain_token, doc in zip(domain_tokens_combined, all_a_documents)
    ]

    modified_b_docs = [
        f"{domain_token}{doc}[SEP]"
        for domain_token, doc in zip(domain_tokens_combined, all_b_documents)
    ]

    # Deduplicate the modified documents while preserving the expert assignment.
    # If the same modified document appears more than once, we check that the expert assignment is consistent.
    doc_to_expert = {}
    for text, expert in zip(modified_docs, expert_assignments_combined):
        if text in doc_to_expert:
            if doc_to_expert[text] != expert:
                print(f"Warning: Duplicate document with differing expert assignments detected for:\n{text}")
        else:
            doc_to_expert[text] = expert

    # Extract the unique texts and their corresponding expert assignments.
    texts = list(doc_to_expert.keys())
    expert_assignments = [doc_to_expert[text] for text in texts]

    # Now, unique_texts contains the deduplicated, domain-token-enhanced texts,
    # and unique_expert_assignments contains the matching expert assignments.


    # This list will collect summary metric records for all model/dataset combinations.
    summary_records = []
    # Loop over each model you want to evaluate.
    for model_name, model_path in MODEL_DICT.items():
        print(f"Processing model: {model_name}")

        # Create a directory for the current model's output.
        model_dir = os.path.join(result_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Check if the aggregated results already exist.
        agg_preds_file = os.path.join(model_dir, f"{model_name}_all_predictions.csv")
        agg_metrics_file = os.path.join(model_dir, f"{model_name}_all_metrics.csv")
        if os.path.exists(agg_preds_file) and os.path.exists(agg_metrics_file):
            print(f"Results for model {model_name} already exist. Skipping recalculation.")
            continue

        if test_mode:
            # Generate random embeddings dictionary
            embeddings_dict = {text: torch.randn(1, 128, dtype=torch.float32) for text in texts}

        else:
            domains = list(DATA_DICT.keys())
            embedder = MoeModernBertEmbedder(model_path, domains).to(DEVICE)
            embeddings_dict = embedder.embed_dataset(
                texts,
                assignments=expert_assignments,
                tokenizer=embedder.tokenizer,
                batch_size=2,
                cls_pooling=False,
                save=True,
                save_path=os.path.join(model_dir, f"{model_name}_embeddings.pth"),
                add_special_tokens=False
            )

        # Prepare dictionaries to hold results per domain and overall.
        result_dict = defaultdict(lambda: {'preds': [], 'labels': []})
        aggregated_preds = []
        aggregated_labels = []
        aggregated_rows = []  # To record domain info with each prediction for later aggregation.

        # Iterate through each evaluation example.
        for a, b, domain_token, label in zip(modified_a_docs, modified_b_docs, all_domain_tokens, all_labels):
            a_emb = embeddings_dict[a]
            b_emb = embeddings_dict[b]
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

        embedder.cpu()
        del embedder
        torch.cuda.empty_cache()

        # Compute and save aggregated metrics (i.e. across all domains).
        aggregated_metrics = compute_metrics_benchmark(aggregated_preds, aggregated_labels)

        # Save aggregated per-example predictions (with domain info).
        df_agg_preds = pd.DataFrame(aggregated_rows)
        df_agg_preds.to_csv(agg_preds_file, index=False)

        # Save aggregated metrics.
        df_agg_metrics = pd.DataFrame([aggregated_metrics])
        df_agg_metrics.to_csv(agg_metrics_file, index=False)

        # Record aggregated summary metrics.
        summary_record = {"Model": model_name, "Dataset": "All"}
        summary_record.update(aggregated_metrics)
        summary_records.append(summary_record)

    # Save the overall summary CSV.
    summary_csv_file = os.path.join(result_dir, "trained_summary.csv")
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Summary saved to {summary_csv_file}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate embedding models and output predictions and metrics to CSV files.")
    parser.add_argument("--result_dir", type=str, default="results", help="Directory to save the CSV output files.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of text to embed")
    parser.add_argument("--test", action="store_true", help="Run in test mode with random embeddings")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    main(args)
