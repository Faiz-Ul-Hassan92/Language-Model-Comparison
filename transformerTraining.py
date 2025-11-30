import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

X_train = np.load('datasetFormatted/X_train.npy')
X_val = np.load('datasetFormatted/X_val.npy')
X_test = np.load('datasetFormatted/X_test.npy')
y_train = np.load('datasetFormatted/y_train.npy')
y_val = np.load('datasetFormatted/y_val.npy')
y_test = np.load('datasetFormatted/y_test.npy')

with open('tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('tokenizer/config.pkl', 'rb') as f:
    config = pickle.load(f)
    vocab_size = config['vocab_size']
    max_sequence_len = config['max_sequence_len']




class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(max_seq_len, d_model)
    
    def positional_encoding(self, max_seq_len, d_model):
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]
    




class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
    
    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    






def build_transformer_model(vocab_size, max_sequence_len, embedding_dim=128, num_heads=4, ff_dim=512, num_blocks=2, dropout=0.2):
    inputs = Input(shape=(max_sequence_len-1,))
    
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = PositionalEncoding(max_sequence_len-1, embedding_dim)(x)
    
    for _ in range(num_blocks):
        x = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)(x)
    
    x = x[:, -1, :] #fixed this from globalAveraging(Was giving bas results) to last token as embedding for generation
    x = Dropout(dropout)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model





def train_model(model, optimizer_name, epochs=30, batch_size=128):
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"\nTraining Transformer with {optimizer_name.upper()} optimizer...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    return history, training_time






def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_probs = model.predict(X_test, verbose=0)
    perplexity = np.exp(loss)
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }





def plot_history(history, optimizer_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Transformer + {optimizer_name.upper()} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Transformer + {optimizer_name.upper()} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'transformer/baseline/transformer_{optimizer_name}_training.png', dpi=150)
    plt.show()





optimizers = ['adam', 'rmsprop', 'sgd']
results = {}

for opt in optimizers:
    print(f"Transformer with {opt.upper()}")
    tf.keras.backend.clear_session()
    model = build_transformer_model(vocab_size, max_sequence_len)
    history, training_time = train_model(model, opt)
    metrics = evaluate_model(model, X_test, y_test)
    
    results[f'Transformer_{opt}'] = {
        'optimizer': opt,
        'training_time': training_time,
        'test_loss': metrics['loss'],
        'test_accuracy': metrics['accuracy'],
        'test_perplexity': metrics['perplexity']
    }
    
    print(f"\nResults for Transformer + {opt.upper()}:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Test Loss: {metrics['loss']:.4f}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test Perplexity: {metrics['perplexity']:.2f}")
    
    plot_history(history, opt)
    
    model.save(f'transformer/baseline/transformer_{opt}_model.keras')
    print(f"Model saved as: transformer_{opt}_model.keras")

for name, res in results.items():
    print(f"{name}: Perplexity={res['test_perplexity']:.2f}, Accuracy={res['test_accuracy']:.4f}, Time={res['training_time']:.2f}s")

with open('transformer/baseline/transformer_results.pkl', 'wb') as f:
    pickle.dump(results, f)
