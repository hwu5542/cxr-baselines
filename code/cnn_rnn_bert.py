import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LSTM, Dense, Embedding, Attention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel, BertTokenizer
from save_and_load_parquet import SaveAndLoadParquet
import logging

# Configure logging (matches ngram.py exactly)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cnn_rnn_bert.log'),
        logging.StreamHandler()
    ]
)

# Configuration (matches ngram.py structure)
DATA_DIR = "D:/mimic/processed"
OUTPUT_DIR = "D:/mimic/outputs/cnn_rnn_bert"
MAX_SEQ_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 64
INIT_LR = 0.001
BEAM_WIDTH = 4

# Initialize tokenizer (for text generation)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
VOCAB_SIZE = tokenizer.vocab_size

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class ImageEncoder(tf.keras.layers.Layer):
    """Encoder with pre-extracted pooled_features"""
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # self.cnn = tf.keras.Sequential([
        #     Conv2D(64, 3, activation='relu', padding='same'),
        #     Conv2D(64, 3, activation='relu', padding='same'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     Conv2D(128, 3, activation='relu', padding='same'),
        #     Conv2D(128, 3, activation='relu', padding='same'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     tf.keras.layers.Flatten(),
        #     Dense(256, activation='relu')
        # ])
        self.projection = Dense(256, activation='relu')
        
    def call(self, features, training=False):
        # return self.cnn(images)
        return self.projection(features)

class ReportDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super(ReportDecoder, self).__init__()
        self.embedding = Embedding(VOCAB_SIZE, 256)
        self.lstm = LSTM(256, return_sequences=True, return_state=True)
        # self.attention = Attention()
        self.dense = Dense(VOCAB_SIZE, activation='softmax')
        
    def call(self, inputs, initial_state=None):
        x, encoder_output = inputs
        x = self.embedding(x)
        lstm_output, state_h, state_c = self.lstm(x, initial_state=initial_state)
        # context_vector = self.attention([lstm_output, encoder_output])
        # combined_output = tf.concat([lstm_output, context_vector], axis=-1)
        # return self.dense(combined_output), [state_h, state_c]
        return self.dense(lstm_output), [state_h, state_c]

class CNN_RNN_BERT_Model(tf.keras.Model):
    def __init__(self):
        super(CNN_RNN_BERT_Model, self).__init__()
        self.image_encoder = ImageEncoder()
        self.decoder = ReportDecoder()
        
    def call(self, inputs, training=False):
        features, decoder_input = inputs
        image_features = self.image_encoder(features, training=training)
        image_features = tf.expand_dims(image_features, 1)
        decoder_output, _ = self.decoder((decoder_input, image_features), training=training)
        return decoder_output

def learning_rate_schedule(epoch, lr):
    """Learning rate decay every 16 epochs as in the paper"""
    if (epoch + 1) % 16 == 0:
        return lr * 0.5
    return lr

def teacher_forcing_schedule(epoch):
    """Gradually reduce teacher forcing probability"""
    return min(0.05 * (epoch // 16), 0.8)

def tokenize_reports(reports):
    """Tokenize reports using BERT tokenizer and add special tokens"""
    tokenized = tokenizer(
        reports.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='tf'
    )
    return tokenized['input_ids'], tf.cast(tokenized['attention_mask'], tf.float32)

def create_dataset(features, input_ids, attention_mask):
    """Create TensorFlow dataset from features and tokenized reports"""
    # Ensure features are float32 tensors
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'features': features,
            'decoder_input': input_ids[:, :-1]  # Shifted right for teacher forcing
        },
        {
            'output': input_ids[:, 1:],  # Shifted left for target
            'mask': attention_mask[:, 1:]  # Mask for loss calculation
        }
    ))
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def beam_search_decode(model, features, beam_width=4, max_length=MAX_SEQ_LENGTH):
    # Encode image
    image_features = model.image_encoder(features)
    image_features = tf.expand_dims(image_features, 1)
    
    # Beam search initialization
    start_token = tokenizer.cls_token_id
    end_token = tokenizer.sep_token_id
    
    beams = [([start_token], 0.0, None)]  # (sequence, score, state)
    
    for _ in range(max_length):
        new_beams = []
        for seq, score, state in beams:
            if seq[-1] == end_token:
                new_beams.append((seq, score, state))
                continue
                
            decoder_input = tf.expand_dims([seq[-1]], 0)
            predictions, new_state = model.decoder((decoder_input, image_features), initial_state=state)
            predictions = tf.squeeze(predictions, 0)
            
            top_k = tf.math.top_k(predictions[-1], k=beam_width)
            for i in range(beam_width):
                token = top_k.indices[i].numpy()
                token_prob = top_k.values[i].numpy()
                new_seq = seq + [token]
                new_score = score - np.log(token_prob + 1e-8)
                new_beams.append((new_seq, new_score, new_state))
        
        beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]
        if all(beam[0][-1] == end_token for beam in beams):
            break
    
    # Length normalization
    scored_beams = [(seq, score / (len(seq)**0.7)) for seq, score, _ in beams]
    best_beam = min(scored_beams, key=lambda x: x[1])
    return tokenizer.decode(best_beam[0], skip_special_tokens=True)

# def prepare_dataset(df):
#     """Prepare dataset in same format as ngram.py"""
#     images = np.stack(df['pooled_features'].values)  # Using pooled_features as image representation
#     study_ids = df['study_id'].values
#     reports = df['findings'].values
#     return images, study_ids, reports
def load_and_prepare_data():
    # Load data
    logging.info("Loading data...")
    sl = SaveAndLoadParquet()
    train_df = sl.load_from_parquet(os.path.join(DATA_DIR, "parsed_train.parquet"))
    test_df = sl.load_from_parquet(os.path.join(DATA_DIR, "parsed_test.parquet"))
    # print(train_df.columns.to_list())
    logging.info(f"Train samples: {len(train_df)}")
    logging.info(f"Test samples: {len(test_df)}")

    train_features = np.stack(train_df['pooled_features'].apply(lambda x: np.array(x, dtype=np.float32)))
    test_features = np.stack(test_df['pooled_features'].apply(lambda x: np.array(x, dtype=np.float32)))

    return train_df, test_df, train_features, test_features

def train_and_predict():
    try:
        # Prepare datasets (same as ngram.py)
        train_df, test_df, train_features, test_features = load_and_prepare_data()
        # train_images, train_study_ids, train_reports = prepare_dataset(train_df)
        # test_images, test_study_ids, test_reports = prepare_dataset(test_df)

        logging.info("Preparing datasets...")
        # Tokenize reports
        train_input_ids, train_attention_mask = tokenize_reports(train_df['findings'])
        test_input_ids, test_attention_mask = tokenize_reports(test_df['findings'])
        
        # Create TensorFlow datasets
        # train_dataset = create_dataset(train_df['pooled_features'].values, train_input_ids, train_attention_mask)
        # test_dataset = create_dataset(test_df['pooled_features'].values, test_input_ids, test_attention_mask)
        train_dataset = create_dataset(train_features, train_input_ids, train_attention_mask)
        test_dataset = create_dataset(test_features, test_input_ids, test_attention_mask)

        # Initialize model
        model = CNN_RNN_BERT_Model()
        optimizer = Adam(INIT_LR)
        
        def _compute_gradients(tape, tensor, var_list):
            grads = tape.gradient(tensor, var_list)
            return [grad if grad is not None else tf.zeros_like(var)
                    for var, grad in zip(var_list, grads)]
        
        # Custom training loop to handle teacher forcing
        @tf.function
        def train_step(batch):
            features = batch[0]['features']
            decoder_input = batch[0]['decoder_input']
            targets = batch[1]['output']
            mask = batch[1]['mask']

            with tf.GradientTape() as tape:
                predictions = model((features, decoder_input), training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    targets, predictions, from_logits=False)
                loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
            
            trainable_vars = model.trainable_variables
            # gradients = tape.gradient(loss, trainable_vars)
            gradients = _compute_gradients(tape, loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            return loss

        # Training loop
        logging.info("Starting training...")
        for epoch in range(EPOCHS):
            current_lr = learning_rate_schedule(epoch, INIT_LR)
            tf.keras.backend.set_value(optimizer.learning_rate, current_lr)
            # teacher_forcing_prob = 1.0 - teacher_forcing_schedule(epoch)
            
            # Training step would go here (omitted for brevity)
            # In practice you would need to tokenize reports and create proper batches

            epoch_loss = []
            for batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                loss = train_step(batch)
                epoch_loss.append(loss)
            
            avg_loss = np.mean(epoch_loss)

            logging.info(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - LR: {current_lr:.5f}")

            # Generate predictions every 8 epochs to monitor progress
            if (epoch + 1) % 8 == 0 or epoch == EPOCHS - 1:
                predictions = []
                for i in tqdm(range(len(test_df)), desc="Generating predictions"):
                    # Get test features and generate report
                    features = tf.expand_dims(test_df['pooled_features'].iloc[i], 0)
                    generated_report = beam_search_decode(model, features, beam_width=BEAM_WIDTH)

                    predictions.append({
                        'study_id': test_df['study_id'].iloc[i],
                        'patient_id': test_df['patient_id'].iloc[i],
                        'true_report': test_df['findings'].iloc[i],
                        'pred_report': generated_report,
                        'epoch': epoch + 1
                    })
                    
                    # Log first 3 samples
                    if i < 3:
                        logging.info(f"\nSample {i+1}:")
                        logging.info(f"True report: {test_df['findings'].iloc[i]}")
                        logging.info(f"Generated report: {generated_report}")

                # Save predictions (same format as ngram.py)
                results_df = pd.DataFrame(predictions)
                Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
                
                # Save in same formats as ngram.py
                output_path = f"{OUTPUT_DIR}/epoch{epoch+1}_predictions.parquet"
                results_df.to_parquet(output_path)
                logging.info(f"Saved predictions to {output_path}")
                
                # # Also save CSV version
                # results_df[['study_id', 'true_report', 'pred_report']].to_csv(
                #     f"{OUTPUT_DIR}/predictions_epoch{epoch+1}.csv", index=False
                # )
        
        logging.info("Training completed successfully")
        return results_df
        
    except Exception as e:
        logging.error(f"Error in CNN-RNN-BERT model: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_and_predict()