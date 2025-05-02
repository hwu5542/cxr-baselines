# evaluate_all.py
import sys
import pandas as pd
from pathlib import Path
from evaluate_chexpert import evaluate_chexpert
from evaluate_nlg import evaluate_nlg_metrics
import os

# Configuration
# 0 1 1nn
# 1 n cnn-rnn, 1 1 cnn-rnn with 8 epoch, ...
# 2 n ngam, 2 1 1-gram, ...
# 3 1 random
INPUT_DIR = Path("./input")
EVAL_DIR = Path("./output")
MODELS = ["1nn", "cnn_rnn_bert", "ngram", "random"]


def load_predictions(model_dir: Path, model_sub):
    """Load predictions for a given model"""
    if model_dir.name == "cnn_rnn_bert":
        # Handle multiple epoch files for cnn_rnn_bert
        if model_sub:
            pred_files = list(
                model_dir.glob(f"epoch{8 * model_sub}_predictions.parquet")
            )
        else:
            pred_files = list(model_dir.glob("epoch*_predictions.parquet"))

        return {f.stem.split("_")[0]: pd.read_parquet(f) for f in pred_files}
    elif model_dir.name == "ngram":
        # Handle ngram variants
        if model_sub:
            target_ngram = f"{model_sub}gram"
        else:
            target_ngram = f"*gram"

        return {
            f.stem.split("_")[0]: pd.read_parquet(f)
            for f in model_dir.glob(f"{target_ngram}_predictions.parquet")
        }
    else:
        # Single prediction file models
        return {"default": pd.read_parquet(model_dir / "predictions.parquet")}


def run_evaluations():
    # Create evaluation directory if not exists
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    model_num = None
    model_sub = None
    test_size = 0

    if len(sys.argv) > 2:
        # partition mode
        model_num = int(sys.argv[1])
        model_sub = int(sys.argv[2])
    elif len(sys.argv) > 1:
        # Test Mode On
        test_size = int(sys.argv[1])

    # Load all model predictions
    for index, model in enumerate(MODELS):
        print(index, model_num)
        model_dir = INPUT_DIR / model
        if not model_dir.exists():
            continue

        if model_num and model_num != index:
            continue

        print(f"\nEvaluating model: {model}")
        predictions = load_predictions(model_dir, model_sub)

        for variant, pred_df in predictions.items():
            print(f"  Processing variant: {variant}")
            output_subdir = EVAL_DIR / model / variant
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Prepare reference reports (true reports)
            if test_size > 0:
                pred_df = pred_df.head(test_size)

            # references.rename(columns={'true_report': 'text'}, inplace=True)
            references = pred_df[["study_id", "true_report"]].copy()

            # Run NLG evaluation (BLEU, CIDEr)
            evaluate_nlg_metrics(
                predictions_df=pred_df,
                references_df=references,
                output_dir=str(output_subdir),
            )

            # Run CheXpert evaluation
            evaluate_chexpert(pred_df, "pred_report", str(output_subdir))

            evaluate_chexpert(
                pred_df, "true_report", str(output_subdir), suffix="_true"
            )

    print("\nAll evaluations completed!")


if __name__ == "__main__":
    run_evaluations()
