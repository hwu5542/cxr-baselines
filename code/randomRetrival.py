import os
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging
from save_and_load_parquet import SaveAndLoadParquet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('random_retrieval.log'),
        logging.StreamHandler()
    ]
)

# Path configuration
train_path = 'D:/mimic/processed/parsed_train.parquet'
test_path = 'D:/mimic/processed/parsed_test.parquet'
output_dir = 'D:/mimic/outputs/random'

class RandomRetrievalBaseline:
    def __init__(self):
        """Initialize random retrieval baseline"""
        self.train_reports = None
        self.study_ids = None
    
    def fit(self, train_df: pd.DataFrame):
        """Store training data for random retrieval
        
        Args:
            train_df: DataFrame containing:
                      - 'findings' (parsed report texts)
                      - 'study_id' (optional)
        """
        logging.info("Loading training data...")
        self.train_reports = train_df['findings'].values
        self.study_ids = train_df['study_id'].values if 'study_id' in train_df.columns else None
        logging.info(f"Loaded {len(self.train_reports)} training samples")
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate random predictions from training set
        
        Args:
            test_df: DataFrame with same structure as train_df
            
        Returns:
            DataFrame with columns: ['study_id', 'true_report', 'pred_report']
        """
        logging.info("Generating random predictions...")
        predictions = []
        
        for i in tqdm(range(len(test_df)), desc="Generating random reports"):
            # Select a random report from training set
            random_idx = random.randint(0, len(self.train_reports) - 1)
            
            predictions.append({
                'study_id': test_df['study_id'].iloc[i] if 'study_id' in test_df.columns else i,
                'true_report': test_df['findings'].iloc[i],
                'pred_report': self.train_reports[random_idx],
                'random_index': random_idx  # Track which training sample was used
            })
            
        return pd.DataFrame(predictions)
    
    def save_predictions(self, predictions: pd.DataFrame, output_dir: str):
        """Save predictions in optimal format for evaluation
        
        Args:
            predictions: DataFrame from predict()
            output_dir: Directory to save outputs
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats:
        # 1. Parquet (compact, preserves data types)
        predictions.to_parquet(f"{output_dir}/predictions.parquet")
        logging.info(f"Saved predictions to {output_dir}/predictions.parquet")
        
        # # 2. CSV (human readable)
        # csv_path = f"{output_dir}/predictions.csv"
        # predictions[['study_id', 'true_report', 'pred_report']].to_csv(csv_path, index=False)
        # logging.info(f"Saved CSV version to {csv_path}")

def run_random_baseline():
    try:
        # Initialize data loader
        sl = SaveAndLoadParquet()
        
        # Load preprocessed data
        logging.info("Loading data...")
        train_df = sl.load_from_parquet(train_path)
        test_df = sl.load_from_parquet(test_path)
        
        logging.info(f"Train samples: {len(train_df)}")
        logging.info(f"Test samples: {len(test_df)}")
        
        # Initialize and run random baseline
        rr = RandomRetrievalBaseline()
        rr.fit(train_df)
        predictions = rr.predict(test_df)
        
        # Save results
        rr.save_predictions(predictions, output_dir)
        logging.info("Random retrieval baseline completed successfully")
        
    except Exception as e:
        logging.error(f"Error in random retrieval baseline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_random_baseline()