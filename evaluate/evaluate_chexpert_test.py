import pandas as pd
import os
import random
from pathlib import Path
from loader import Loader
from stages import Extractor, Classifier, Aggregator

def evaluate_chexpert():
    """Evaluate generated reports using CheXpert labeler
    
    Args:
        predictions_df: DataFrame containing 'study_id', 'true_report', 'pred_report'
        output_dir: Directory to save evaluation results
    """
    
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

if __name__ == "__main__":
    evaluate_chexpert()