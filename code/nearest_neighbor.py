import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm
import pickle
from save_and_load_parquet import SaveAndLoadParquet

train_path = 'D:/mimic/processed/parsed_train.parquet'
test_path = 'D:/mimic/processed/parsed_test.parquet'
output_dir = 'D:/mimic/outputs/1nn'

class NearestNeighborBaseline:
    def __init__(self, k=1, feature_type='pooled'):
        """Initialize nearest neighbor baseline
        
        Args:
            k: Number of neighbors to consider
            feature_type: Type of features to use ('pooled' or 'spatial')
                         'pooled': 1024-D global features
                         'spatial': 1024x8x8 spatial features (will be average pooled)
        """
        self.k = k
        self.feature_type = feature_type
        self.train_features = None
        self.train_reports = None
        self.study_ids = None
    
    def fit(self, train_df: pd.DataFrame):
        """Store training data for nearest neighbor search
        
        Args:
            train_df: DataFrame containing:
                      - 'pooled_features' or 'spatial_features'
                      - 'findings' (parsed report texts)
                      - 'study_id' (optional)
        """
        if self.feature_type == 'pooled':
            self.train_features = np.stack(train_df['pooled_features'].values)
        elif self.feature_type == 'spatial':
            # Average pool spatial features to get global representation
            spatial_features = np.stack(train_df['spatial_features'].values)
            self.train_features = spatial_features.mean(axis=(2, 3))  # [n, 1024]
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
            
        self.train_reports = train_df['findings'].values
        self.study_ids = train_df['study_id'].values if 'study_id' in train_df.columns else None
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Find nearest neighbors and return predictions
        
        Args:
            test_df: DataFrame with same structure as train_df
            
        Returns:
            DataFrame with columns: ['study_id', 'true_report', 'pred_report', 'neighbor_indices']
        """
        # Extract features from test data
        if self.feature_type == 'pooled':
            test_features = np.stack(test_df['pooled_features'].values)
        else:
            spatial_features = np.stack(test_df['spatial_features'].values)
            test_features = spatial_features.mean(axis=(2, 3))
        
        # Compute similarities
        sim_matrix = cosine_similarity(test_features, self.train_features)
        neighbor_indices = np.argpartition(sim_matrix, -self.k, axis=1)[:, -self.k:]
        
        # Get predictions
        predictions = []
        for i in tqdm(range(len(test_features)), desc="Finding neighbors"):
            top_idx = neighbor_indices[i][0]  # For k=1
            predictions.append({
                'study_id': test_df['study_id'].iloc[i] if 'study_id' in test_df.columns else i,
                'true_report': test_df['findings'].iloc[i],
                'pred_report': self.train_reports[top_idx],
                'neighbor_indices': neighbor_indices[i],
                'similarity_score': sim_matrix[i, top_idx]
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
        
        # 2. CSV (human readable)
        predictions[['study_id', 'true_report', 'pred_report', 'similarity_score']].to_csv(
            f"{output_dir}/predictions.csv", index=False
        )
        
        # 3. Pickle (preserves exact Python objects)
        with open(f"{output_dir}/neighbor_indices.pkl", 'wb') as f:
            pickle.dump(predictions['neighbor_indices'].values, f)

def run_nn_baseline(feature_type='pooled'):
    # Load preprocessed data with features
    sl = SaveAndLoadParquet()
    train_df = sl.load_from_parquet(train_path)
    test_df = sl.load_from_parquet(test_path)
    
    # Initialize and run NN baseline
    nn = NearestNeighborBaseline(k=1, feature_type=feature_type)
    nn.fit(train_df)
    predictions = nn.predict(test_df)
    
    # Save results
    nn.save_predictions(predictions, output_dir)
    print(f"Saved predictions to {output_dir}")

if __name__ == "__main__":
    # Can choose between 'pooled' (1024-D) or 'spatial' (1024x8x8) features
    run_nn_baseline(feature_type='pooled')