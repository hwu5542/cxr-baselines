import os
import random
import pandas as pd
from load_and_preprocess_data_observable import ObservableDataProcessor
from tqdm import tqdm
from report_parser import parse_report

train_path = 'D:/mimic/processed/train.parquet'
test_path = 'D:/mimic/processed/test.parquet'
output_dir = 'D:/mimic/outputs/random'

def random_retrieve(reports, n_samples=1):
    """Randomly select reports from training set"""
    return random.sample(reports, n_samples)

def main():

    # Initialize Data processor
    odp = ObservableDataProcessor()

    # Load data
    train_df = odp.load_from_parquet(train_path)
    test_df = odp.load_from_parquet(test_path)

    # Get reports
    # Parse reports (if not already parsed during preprocessing)
    train_df['findings'] = train_df['report'].apply(lambda x: parse_report(x)['findings'])
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
    pd.DataFrame.from_dict(predictions, orient='index').to_json('outputs/random/predictions.json')

if __name__ == "__main__":
    main()