import pandas as pd
import numpy as np

class SaveAndLoadParquet:
    def save_to_parquet(self, df, output_path):
        # self.current_stage = "Saving Data"
        # self.log_progress(f"Saving to {output_path}...")
        
        # Convert numpy arrays to bytes
        df = df.copy()
        if 'image' in df.columns:
            df['image'] = df['image'].apply(
                lambda x: x.tobytes() if isinstance(x, np.ndarray) else x
            )
        
        df.to_parquet(output_path)
        # self.log_progress(f"Saved {len(df)} items to {output_path}")

    def load_from_parquet(self, file_path):
        # self.current_stage = "Loading Data"
        # self.log_progress(f"Loading from {file_path}...")
        
        df = pd.read_parquet(file_path)
        
        # Convert bytes back to numpy arrays
        if 'image' in df.columns:
            df['image'] = df['image'].apply(
                lambda x: np.frombuffer(x, dtype=np.float32).reshape(224, 224) 
                if isinstance(x, bytes) else x
            )
        
        # self.log_progress(f"Loaded {len(df)} items from {file_path}")
        return df