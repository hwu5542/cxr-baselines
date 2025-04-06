import os
import random
import pandas as pd
from tqdm import tqdm
from save_and_load_parquet import SaveAndLoadParquet

train_path = 'D:/mimic/processed/parsed_train.parquet'
test_path = 'D:/mimic/processed/parsed_test.parquet'
output_dir = 'D:/mimic/outputs/random'

def random_retrieve(reports, n_samples=1):
    """Randomly select reports from training set"""
    return random.sample(reports, n_samples)

def main():

    # Initialize Data processor
    sl = SaveAndLoadParquet()

    # Load data
    train_df = sl.load_from_parquet(train_path)
    test_df = sl.load_from_parquet(test_path)

    print(train_df.shape)
    print(test_df.shape)
    # Get reports
    # Parse reports (if not already parsed during preprocessing)    
    train_reports = train_df['findings'].tolist()
    test_ids = test_df['study_id'].tolist()
    
    # Generate predictions
    predictions = {}
    for test_id in tqdm(test_ids):
        pred_report = random_retrieve(train_reports)[0]
        predictions[test_id] = pred_report
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results
    pd.DataFrame.from_dict(predictions, orient='index').to_json(os.path.join(output_dir, 'predictions.json'))

if __name__ == "__main__":
    main()