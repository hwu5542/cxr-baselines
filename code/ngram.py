import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from save_and_load_parquet import SaveAndLoadParquet
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ngram.log'),
        logging.StreamHandler()
    ]
)

# Configuration
DATA_DIR = "D:/mimic/processed"
REPORTS_DIR = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/reports/files"
N_NEIGHBORS = 100  # Number of neighbors to consider for conditional n-gram
N_GRAM = 3  # n-gram order
START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'  # For unknown words
OUTPUT_DIR = "D:/mimic/outputs/ngram"

class ConditionalNGramModel:
    def __init__(self, n=3):
        self.n = n
        self.vocab = set()
        self.lm = defaultdict(Counter)
        
    def train(self, reports):
        """Train n-gram language model on given reports"""
        for report in reports:
            if not isinstance(report, str):
                continue
                
            # Tokenize and add start/end tokens
            tokens = report.lower().split()
            padded_tokens = [START_TOKEN] * (self.n-1) + tokens + [END_TOKEN]
            
            # Update vocabulary
            self.vocab.update(tokens)

            # Count n-gram occurrences
            for i in range(len(padded_tokens) - self.n + 1):
                context = tuple(padded_tokens[i:i+self.n-1]) if self.n > 1 else ()
                target = padded_tokens[i+self.n-1]
                self.lm[context][target] += 1
                
    def generate(self, seed=None, max_length=100, temperature=1.0):
        """Generate text from the language model
        
        Args:
            seed: Optional seed text to start generation
            max_length: Maximum length of generated text
            temperature: Controls randomness (1.0=normal, <1.0=more conservative)
            
        Returns:
            Generated text string
        """
        if seed is None:
            current = [START_TOKEN] * (self.n-1)
        else:
            seed_tokens = seed.lower().split()
            current = ([START_TOKEN] * (self.n-1 - len(seed_tokens))) + seed_tokens
            current = current[-(self.n-1):]  # Truncate to context size
            
        # current = list(seed)
        result = current.copy()
        
        while len(result) < max_length + (self.n-1):
            context = tuple(current[-(self.n-1):]) if self.n > 1 else ()
            
            if context not in self.lm:
                break

            # Get possible next words and their counts
            words, counts = zip(*self.lm[context].items())
            
            # Apply temperature
            counts = np.array(counts) ** (1/temperature)
            probs = counts / sum(counts)
            
            # Sample next word
            next_word = np.random.choice(words, p=probs)
            
            if next_word == END_TOKEN:
                break
                
            result.append(next_word)
            current.append(next_word)
            
        # Remove start tokens from result
        return ' '.join(result[self.n-1:])

# def get_nearest_neighbors(query_feature, train_features, train_study_ids, k=100):
#     """Find k nearest neighbors based on cosine similarity"""
#     # print(query)
#     similarities = cosine_similarity([query_feature], train_features)[0]
#     top_indices = np.argsort(similarities)[-k:]
#     return train_study_ids[top_indices]

def run_ngram_model(n_gram=3):
    """Run conditional n-gram model with specified n-gram order
    
    Args:
        n_gram: Order of n-gram model (1, 2, or 3)
    """
    try:
        logging.info(f"Initializing {n_gram}-gram model...")

        logging.info("Loading data...")
        # Load the parsed data with features
        sl = SaveAndLoadParquet()
        train_df = sl.load_from_parquet(os.path.join(DATA_DIR, "parsed_train.parquet"))
        test_df = sl.load_from_parquet(os.path.join(DATA_DIR, "parsed_test.parquet"))
        
        logging.info(f"Train samples: {len(train_df)}")
        logging.info(f"Test samples: {len(test_df)}")

        # Extract features and reports
        train_features = np.stack(train_df['pooled_features'].values)
        train_reports = train_df['findings'].values
        test_features = np.stack(test_df['pooled_features'].values)
        
        # # Build study_id to report lookup
        # report_lookup = dict(zip(train_df['study_id'], train_df['findings']))
        
        # Prepare test set generation
        # test_study_ids = test_df['study_id'].values
        # train_study_ids = train_df['study_id'].values

        # Compute similarities once for all test samples
        logging.info("Computing similarities...")
        sim_matrix = cosine_similarity(test_features, train_features)
        neighbor_indices = np.argpartition(sim_matrix, -N_NEIGHBORS, axis=1)[:, -N_NEIGHBORS:]

        # Generate reports for test set using conditional n-gram
        # generated_reports = []
        # for test_feature, test_study_id in tqdm(zip(test_features, test_study_ids), 
        #                                       total=len(test_features), 
        #                                       desc="Generating reports"):
        #     # Find nearest neighbors
        #     neighbor_study_ids = get_nearest_neighbors(test_feature, train_features, train_study_ids, N_NEIGHBORS)
        #     neighbor_reports = [report_lookup[sid] for sid in neighbor_study_ids if sid in report_lookup]
            
        #     # Train n-gram model on neighbors' reports
        #     ngram_model = ConditionalNGramModel(n=N_GRAM)
        #     ngram_model.train(neighbor_reports)
            
        #     # Generate report
        #     generated_report = ngram_model.generate(max_length=100)
        #     generated_reports.append({
        #         'study_id': test_study_id,
        #         'generated_report': generated_report
        #     })
        predictions = []
        for i in tqdm(range(len(test_features)), desc=f"Generating {n_gram}-gram reports"):    
            # Get neighbor indices and reports
            neighbors = neighbor_indices[i]
            neighbor_reports = [train_reports[idx] for idx in neighbors if isinstance(train_reports[idx], str)]
            
            # Train n-gram model on neighbors' reports
            ngram_model = ConditionalNGramModel(n=n_gram)
            ngram_model.train(neighbor_reports)
            
            # Generate report
            generated_report = ngram_model.generate(max_length=100)
            
            predictions.append({
                'study_id': test_df['study_id'].iloc[i],
                'true_report': test_df['findings'].iloc[i],
                'pred_report': generated_report,
                'neighbor_indices': neighbors,
                'similarity_score': sim_matrix[i, neighbors[0]] # Score with top neighbor
            })

        # # Convert to DataFrame and save
        # results_df = pd.DataFrame(generated_reports)

        # Convert to DataFrame
        results_df = pd.DataFrame(predictions)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # 1. Parquet (main format)
        output_path = f"{OUTPUT_DIR}/{n_gram}gram_predictions.parquet"
        results_df.to_parquet(output_path)
        logging.info(f"Saved predictions to {output_path}")

        # # Save results
        # Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        # output_path = os.path.join(OUTPUT_DIR, f"{N_GRAM}-gram.tsv")
        # results_df.to_csv(output_path, sep='\t', index=False)
        logging.info(f"{n_gram}-gram model completed successfully")
        
    except Exception as e:
        logging.error(f"Error in n-gram model: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Run all n-gram versions (1, 2, and 3)
    for n in [1, 2, 3]:
        run_ngram_model(n_gram=n)