# evaluate_all.py
import pandas as pd
from pathlib import Path
from evaluate_chexpert import evaluate_chexpert
from evaluate_nlg import evaluate_nlg_metrics
import os

# Configuration
INPUT_DIR = Path("D:/mimic/outputs")
EVAL_DIR = Path("D:/mimic/evaluation")
MODELS = ["1nn", "cnn_rnn_bert", "ngram", "random"]

def load_predictions(model_dir: Path):
    """Load predictions for a given model"""
    if model_dir.name == "cnn_rnn_bert":
        # Handle multiple epoch files for cnn_rnn_bert
        pred_files = list(model_dir.glob("epoch*_predictions.parquet"))
        return {f.stem.split('_')[0]: pd.read_parquet(f) for f in pred_files}
    elif model_dir.name == "ngram":
        # Handle ngram variants
        return {f.stem.split('_')[0]: pd.read_parquet(f) for f in model_dir.glob("*gram_predictions.parquet")}
    else:
        # Single prediction file models
        return {"default": pd.read_parquet(model_dir / "predictions.parquet")}

def run_evaluations():
    # Create evaluation directory if not exists
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all model predictions
    for model in MODELS:
        model_dir = INPUT_DIR / model
        if not model_dir.exists():
            continue
            
        print(f"\nEvaluating model: {model}")
        predictions = load_predictions(model_dir)
        
        for variant, pred_df in predictions.items():
            print(f"  Processing variant: {variant}")
            output_subdir = EVAL_DIR / model / variant
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Prepare reference reports (true reports)
            references = pred_df[['study_id', 'true_report']].copy()
            # references.rename(columns={'true_report': 'text'}, inplace=True)
            
            # Run NLG evaluation (BLEU, CIDEr)
            evaluate_nlg_metrics(
                predictions_df=pred_df,
                references_df=references,
                output_dir=str(output_subdir)
            )
            
            # Run CheXpert evaluation
            evaluate_chexpert(
                predictions_df=pred_df,
                output_dir=str(output_subdir)
            )
            
    print("\nAll evaluations completed!")

if __name__ == "__main__":
    run_evaluations()