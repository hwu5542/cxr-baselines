import csv
import numpy as np
import pandas as pd
import os
import random
from pathlib import Path
from loader import Loader
from stages import Extractor, Classifier, Aggregator


def evaluate_chexpert(predictions_df: pd.DataFrame, report_text, output_dir, suffix=""):
    # Initialize CheXpert components
    extractor = Extractor(
        Path("negbio/chexpert/phrases/mention"),
        Path("negbio/chexpert/phrases/unmention"),
        False,
    )

    classifier = Classifier(
        "negbio/chexpert/patterns/pre_negation_uncertainty.txt",
        "negbio/chexpert/patterns/negation.txt",
        "negbio/chexpert/patterns/post_negation_uncertainty.txt",
        verbose=True,
    )

    CATEGORIES = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Lesion",
        "Lung Opacity",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    aggregator = Aggregator(CATEGORIES, False)

    """Helper function to process reports through CheXpert pipeline"""
    tempname = f"/tmp/chexpert-reports-{random.randint(0,10**6)}{suffix}.csv"
    predictions_df[report_text].to_csv(
        tempname, index=False, header=False, quoting=csv.QUOTE_ALL
    )

    loader = Loader(tempname, False, False)
    loader.load()
    extractor.extract(loader.collection)
    classifier.classify(loader.collection)
    labels = aggregator.aggregate(loader.collection)
    os.remove(tempname)

    labels_df = pd.DataFrame(labels, columns=CATEGORIES)
    labels_df.to_csv(f"{output_dir}/chexpert_labels{suffix}.csv", index=False)
