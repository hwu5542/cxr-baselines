import pandas as pd
import numpy as np

class SaveAndLoadParquet:
    def save_to_parquet(self, df, output_path):
        # self.current_stage = "Saving Data"
        # self.log_progress(f"Saving to {output_path}...")
        
        # Convert numpy arrays to bytes
        df = df.copy()

        # Convert numpy arrays to bytes for specified columns
        array_columns = ['image', 'pooled_features']
        
        for col in array_columns:
            if col in df.columns:
                # if col == 'spatial_features':
                #     # For spatial features (3D arrays), we need to store shape info
                #     df[col] = df[col].apply(
                #         lambda x: (x.shape, x.tobytes()) if isinstance(x, np.ndarray) else x
                #     )
                # else:

                # For 1D and 2D arrays
                df[col] = df[col].apply(
                    lambda x: x.tobytes() if isinstance(x, np.ndarray) else x
                )
        
        df.to_parquet(output_path)
        # self.log_progress(f"Saved {len(df)} items to {output_path}")

    def load_from_parquet(self, file_path):
        # self.current_stage = "Loading Data"
        # self.log_progress(f"Loading from {file_path}...")
        
        df = pd.read_parquet(file_path)

        # Convert bytes back to numpy arrays for specified columns
        array_columns = ['image', 'pooled_features']

        # Convert bytes back to numpy arrays
        for col in array_columns:
            if col in df.columns:
                # if col == 'spatial_features':
                #     # For spatial features, we stored shape info
                #     df[col] = df[col].apply(
                #         lambda x: np.frombuffer(x[1], dtype=np.float32).reshape(x[0]) 
                #         if isinstance(x, tuple) else x
                #     )
                if col == 'image':
                    # For image (2D array)
                    df[col] = df[col].apply(
                        lambda x: np.frombuffer(x, dtype=np.float32).reshape(224, 224) 
                        if isinstance(x, bytes) else x
                    )
                else:
                    # For pooled features (1D array)
                    df[col] = df[col].apply(
                        lambda x: np.frombuffer(x, dtype=np.float32) 
                        if isinstance(x, bytes) else x
                    )
        
        # self.log_progress(f"Loaded {len(df)} items from {file_path}")
        return df