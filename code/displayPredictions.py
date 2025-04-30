
import logging
import os
import sys
import pandas as pd

from pathlib import Path
from save_and_load_parquet import SaveAndLoadParquet


DATA_DIR = Path("D:/mimic/outputs")
MODELS = ["1nn", "cnn_rnn_bert", "ngram", "random"]
display = None

class DisplayPredictions:
    def load_predictions(self, model_dir: Path):
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
    
    def displayPredictions(self):

        for index, model in enumerate(MODELS):
            model_dir = DATA_DIR / model
            if not model_dir.exists():
                continue
                
            print(f"\nDisplaying model: {model}")
            predictions = self.load_predictions(model_dir)
            
            for variant, pred_df in predictions.items():
                print(f"  Printing variant: {variant}")
                print(pred_df.columns.to_list())
                print(pred_df[['true_report', 'pred_report']].head())
        

if __name__ == "__main__":
    display = DisplayPredictions()
    display.displayPredictions()