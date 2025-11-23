import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
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





def build_lstm_model(vocab_size, max_sequence_len, embedding_dim=128, rnn_units=128, dropout=0.2):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        LSTM(rnn_units, return_sequences=True),
        LSTM(rnn_units, return_sequences=False),
        Dropout(dropout),
        Dense(vocab_size, activation='softmax')
    ])
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
    
    print(f"\nTraining LSTM with {optimizer_name.upper()} optimizer...")
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
    ax1.set_title(f'LSTM + {optimizer_name.upper()} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'LSTM + {optimizer_name.upper()} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'lstm/baseline/lstm_{optimizer_name}_training.png', dpi=150)
    plt.show()




optimizers = ['adam', 'rmsprop', 'sgd']
results = {}

for opt in optimizers:
    print(f"LSTM with {opt.upper()}")
    
    model = build_lstm_model(vocab_size, max_sequence_len)
    history, training_time = train_model(model, opt)
    metrics = evaluate_model(model, X_test, y_test)
    
    results[f'LSTM_{opt}'] = {
        'optimizer': opt,
        'training_time': training_time,
        'test_loss': metrics['loss'],
        'test_accuracy': metrics['accuracy'],
        'test_perplexity': metrics['perplexity']
    }
    
    print(f"\nResults for LSTM + {opt.upper()}:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Test Loss: {metrics['loss']:.4f}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test Perplexity: {metrics['perplexity']:.2f}")
    
    plot_history(history, opt)
    
    model.save(f'lstm/baseline/lstm_{opt}_model.keras')
    print(f"Model saved as: lstm_{opt}_model.keras")



for name, res in results.items():
    print(f"{name}: Perplexity={res['test_perplexity']:.2f}, Accuracy={res['test_accuracy']:.4f}, Time={res['training_time']:.2f}s")

with open('lstm/baseline/lstm_results.pkl', 'wb') as f:
    pickle.dump(results, f)