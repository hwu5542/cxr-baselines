# evaluate_nlg.py
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pathlib import Path

def evaluate_nlg_metrics(predictions_df: pd.DataFrame, references_df: pd.DataFrame, output_dir: str):
    """Evaluate generated reports using NLG metrics (BLEU, CIDEr)
    
    Args:
        predictions_df: DataFrame containing 'study_id' and 'pred_report'
        references_df: DataFrame containing 'study_id' and 'true_report'
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(references_df.columns.to_list())
    # Prepare data for evaluation
    references = {
        str(row['study_id']): [row['true_report']] 
        for _, row in references_df.iterrows()
    }
    
    predictions = {
        str(row['study_id']): [row['pred_report']] 
        for _, row in predictions_df.iterrows()
    }
    
    # Compute BLEU scores
    bleu_scorer = Bleu(4)
    bleu_scores = bleu_scorer.compute_score(references, predictions)
    
    # Compute CIDEr score
    cider_scorer = Cider()
    cider_score = cider_scorer.compute_score(references, predictions)
    
    # Save results
    results = {
        'bleu_1': bleu_scores[0][0],
        'bleu_2': bleu_scores[0][1],
        'bleu_3': bleu_scores[0][2],
        'bleu_4': bleu_scores[0][3],
        'cider': cider_score[0]
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{output_dir}/nlg_metrics.csv", index=False)