import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class CNNRNNModel:
    def __init__(self, vocab_size, max_seq_length, embedding_dim=256, lstm_units=256):
        """
        Initialize the CNN-RNN model for radiology report generation
        
        Args:
            vocab_size (int): Size of the vocabulary
            max_seq_length (int): Maximum length of the report sequences
            embedding_dim (int): Dimension of the word embeddings
            lstm_units (int): Number of units in the LSTM layer
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        """Build the CNN-RNN model architecture"""
        
        # Image encoder (DenseNet121)
        image_input = Input(shape=(None, None, 3), name='image_input')
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=image_input,
            pooling='avg'
        )
        
        # Freeze the CNN layers (optional - could fine-tune later)
        for layer in base_model.layers:
            layer.trainable = False
            
        # Project CNN features to lower dimension (1024 -> 256)
        cnn_features = base_model.output
        projected_features = Dense(self.embedding_dim, activation='relu', name='feature_projection')(cnn_features)
        
        # Sequence decoder
        caption_input = Input(shape=(self.max_seq_length,), name='caption_input')
        
        # Word embedding layer
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='word_embedding'
        )(caption_input)
        
        # LSTM decoder
        lstm_layer = LSTM(
            self.lstm_units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder'
        )
        
        # Initial state comes from CNN features
        initial_state = [projected_features, projected_features]  # h and c states
        
        # Pass embeddings through LSTM
        lstm_output, _, _ = lstm_layer(embedding_layer, initial_state=initial_state)
        
        # Output layer
        output = Dense(self.vocab_size, activation='softmax', name='output')(lstm_output)
        
        # Define the training model
        self.model = Model(
            inputs=[image_input, caption_input],
            outputs=output,
            name='cnn_rnn_model'
        )
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, train_data, val_data, epochs=64, batch_size=32):
        """
        Train the model with teacher forcing
        
        Args:
            train_data: Training data generator yielding (images, captions, targets)
            val_data: Validation data generator
            epochs: Number of training epochs
            batch_size: Batch size
        """
        # Learning rate schedule (decay by 0.5 every 16 epochs)
        def lr_scheduler(epoch, lr):
            if epoch > 0 and epoch % 16 == 0:
                return lr * 0.5
            return lr
        
        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def predict_beam_search(self, image, tokenizer, beam_width=4, max_length=100):
        """
        Generate report using beam search
        
        Args:
            image: Preprocessed input image
            tokenizer: Tokenizer for converting words to indices and back
            beam_width: Number of beams to keep
            max_length: Maximum length of generated report
            
        Returns:
            Generated report string
        """
        # Get CNN features
        cnn_features = self.model.get_layer('feature_projection').predict(image[np.newaxis, ...])
        
        # Initialize beam search
        start_token = tokenizer.word_index['<start>']
        end_token = tokenizer.word_index['<end>']
        
        # Initialize beams: (sequence, probability)
        beams = [([start_token], 1.0)]
        
        for _ in range(max_length):
            new_beams = []
            
            for seq, prob in beams:
                if seq[-1] == end_token:
                    new_beams.append((seq, prob))
                    continue
                
                # Predict next word probabilities
                input_seq = np.array([seq])
                lstm_output = self.model.get_layer('lstm_decoder')(
                    input_seq,
                    initial_state=[cnn_features, cnn_features]
                )[0]
                word_probs = self.model.get_layer('output')(lstm_output)[0, -1, :]
                
                # Get top k words
                top_k = np.argsort(word_probs)[-beam_width:]
                
                for word_idx in top_k:
                    new_seq = seq + [word_idx]
                    new_prob = prob * word_probs[word_idx]
                    new_beams.append((new_seq, new_prob))
            
            # Select top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if all beams have ended
            if all(seq[-1] == end_token for seq, _ in beams):
                break
        
        # Select best beam
        best_seq = beams[0][0]
        
        # Convert indices to words
        report = ' '.join(tokenizer.index_word[idx] for idx in best_seq if idx not in [start_token, end_token])
        
        return report