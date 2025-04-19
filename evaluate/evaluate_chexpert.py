import csv
import numpy as np
import pandas as pd
import os
import random
from pathlib import Path
from loader import Loader
from stages import Extractor, Classifier, Aggregator

def evaluate_chexpert(predictions_df: pd.DataFrame, output_dir: str):
    """Evaluate generated reports using CheXpert labeler
    
    Args:
        predictions_df: DataFrame containing 'study_id', 'true_report', 'pred_report'
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Write reports to temporary file
    tempname = f'/tmp/chexpert-reports-{random.randint(0,10**6)}.csv'
    predictions_df['pred_report'].to_csv(tempname, index=False, header=False, quoting=csv.QUOTE_ALL)
    
    # Initialize CheXpert components
    extractor = Extractor(
        Path('negbio/chexpert/phrases/mention'), 
        Path('negbio/chexpert/phrases/unmention'), 
        False
    )

    classifier = Classifier(
        'negbio/chexpert/patterns/pre_negation_uncertainty.txt',
        'negbio/chexpert/patterns/negation.txt',
        'negbio/chexpert/patterns/post_negation_uncertainty.txt',
        verbose=True
    )
    
    CATEGORIES = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ]
    
    aggregator = Aggregator(CATEGORIES, False)
    
    # Process reports
    loader = Loader(tempname, False, False)
    loader.load()
    extractor.extract(loader.collection)
    classifier.classify(loader.collection)
    labels = aggregator.aggregate(loader.collection)
    
    # Save results
    if isinstance(labels, np.ndarray):
        labels_df = pd.DataFrame(labels, columns=CATEGORIES)
    else:
        labels_df = labels  # assuming it's already a DataFrame
    
    # Save results
    labels_df.to_csv(f"{output_dir}/chexpert_labels.csv", index=False)

    # Clean up
    os.remove(tempname)