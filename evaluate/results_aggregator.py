# results_aggregator.py
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np

# Configuration
EVAL_DIR = Path("./evaluate/output")
MODELS = ["1nn", "cnn_rnn_bert", "ngram", "random"]
OUTPUT_FILE = "final_results.csv"
F1_TABLE_FILE = "f1_scores_table.csv"


def calculate_chexpert_metrics(true_labels, pred_labels):
    """Calculate accuracy, precision, and F1 for CheXpert results"""
    # Convert to binary classification (1 if any finding, 0 if No Finding)
    true_binary = (true_labels.drop("No Finding", axis=1) == 1).any(axis=1).astype(int)
    pred_binary = (pred_labels.drop("No Finding", axis=1) == 1).any(axis=1).astype(int)

    return {
        "CheXpert Accuracy": accuracy_score(true_binary, pred_binary),
        "CheXpert Precision": precision_score(
            true_binary, pred_binary, zero_division=0
        ),
        "CheXpert F1": f1_score(true_binary, pred_binary, zero_division=0),
    }


def calculate_per_class_f1(true_labels, pred_labels, categories):
    """Calculate per-class F1 scores and averages"""
    results = {}

    # Convert to binary (1 for present, 0 for absent/not mentioned)
    true_binary = (true_labels == 1).astype(int)
    pred_binary = (pred_labels == 1).astype(int)

    # adding the model and variant in first and second columns
    results["Model"] = None
    results["Variant"] = None

    # Calculate per-class F1
    for i, category in enumerate(categories):
        f1 = f1_score(true_binary[:, i], pred_binary[:, i], zero_division=0)
        results[category] = f1

    # Calculate micro-average (global F1)
    micro_f1 = f1_score(true_binary.ravel(), pred_binary.ravel(), zero_division=0)
    results["Micro-Average"] = micro_f1

    # Calculate macro-average (mean of per-class F1)
    macro_f1 = np.mean([results[cat] for cat in categories])
    results["Macro-Average"] = macro_f1

    return results


def aggregate_results():
    all_results = []
    f1_scores_data = []

    CATEGORIES = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Lesion",
        "Lung Opacity",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    for model in MODELS:
        model_dir = EVAL_DIR / model
        if not model_dir.exists():
            continue

        variants = [d.name for d in model_dir.iterdir() if d.is_dir()]

        for variant in variants:
            variant_dir = model_dir / variant

            # Load NLG metrics
            nlg_file = variant_dir / "nlg_metrics.csv"
            if nlg_file.exists():
                nlg_metrics = pd.read_csv(nlg_file).iloc[0].to_dict()
            else:
                nlg_metrics = {
                    "bleu_1": np.nan,
                    "bleu_2": np.nan,
                    "bleu_3": np.nan,
                    "bleu_4": np.nan,
                    "cider": np.nan,
                }

            # Load CheXpert labels
            true_labels = pd.read_csv(variant_dir / "chexpert_labels_true.csv")
            pred_labels = pd.read_csv(variant_dir / "chexpert_labels.csv")

            # Calculate overall metrics
            chexpert_metrics = calculate_chexpert_metrics(true_labels, pred_labels)

            # Calculate per-class F1 scores
            f1_scores = calculate_per_class_f1(
                true_labels.values, pred_labels.values, CATEGORIES
            )

            # Store F1 scores for the detailed table
            f1_scores["Model"] = model
            f1_scores["Variant"] = variant
            f1_scores_data.append(f1_scores)

            # Combine results for main table
            result = {
                "Model": model,
                "Variant": variant,
                **nlg_metrics,
                **chexpert_metrics,
            }
            all_results.append(result)

    # Create and save main results table
    results_df = pd.DataFrame(all_results)
    results_df = results_df.round(3)
    results_df.replace(0, "< 0.001", inplace=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Create and save F1 scores table
    f1_df = pd.DataFrame(f1_scores_data)
    f1_df = f1_df.round(3)
    f1_df.replace(0, "< 0.001", inplace=True)
    f1_df.to_csv(F1_TABLE_FILE, index=False)

    print(f"Main results saved to {OUTPUT_FILE}")
    print(f"Per-class F1 scores saved to {F1_TABLE_FILE}")
    print("\nMain Results Table:")
    print(results_df)
    print("\nF1 Scores Table:")
    print(f1_df)


if __name__ == "__main__":
    aggregate_results()
