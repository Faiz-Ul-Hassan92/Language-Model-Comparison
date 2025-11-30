## Overview
Comparative analysis of sequence models (RNN, LSTM, Transformer) with optimization 
algorithms (Adam, RMSprop, SGD) for Urdu poetry text generation.

Dataset: ReySajju742/Urdu-Poetry-Dataset (1,323 poems, classical Urdu poetry)
Task: Next-word prediction using n-gram sequences

---

## Project Structure

├── datasetFormatted/          # Preprocessed train/val/test splits (80-10-10)
├── tokenizer/                 # Keras tokenizer & config
├── rnn/baseline/              # RNN experiments
├── lstm/baseline/             # LSTM experiments  
├── transformer/baseline/      # Transformer experiments
├── preprocess.py              # Data extraction & n-gram creation
├── train_rnn.py               # RNN training (3 optimizers)
├── train_lstm.py              # LSTM training (3 optimizers)
├── train_transformer.py       # Transformer training (3 optimizers)
└── generate_*.py              # Text generation scripts

---

## Experiments Conducted

### Baseline Models (9 combinations total), baselines, and further experiments 
┌─────────────┬──────┬─────────┬─────┐
│ Architecture│ Adam │ RMSprop │ SGD │
├─────────────┼──────┼─────────┼─────┤
│ RNN         │  ✓   │    ✓    │  ✓  │
│ LSTM        │  ✓   │    ✓    │  ✓  │
│ Transformer │  ✓   │    ✓    │  ✓  │
└─────────────┴──────┴─────────┴─────┘

### Configuration
- Vocabulary: ~10,500 Urdu tokens
- Sequence length: 20 tokens (capped)
- Batch size: 128
- Early stopping: patience=5
- Dropout: 0.2
- Embedding dim: 128

### Architecture Details
**RNN**: 2-layer SimpleRNN (128 units each)
**LSTM**: 2-layer LSTM (128 units each)  
**Transformer**: 4 attention heads, 2 blocks, FFN=512

### Learning Rates
- Adam/RMSprop: 0.001
- SGD: 0.01 (momentum=0.9)

---

## Text Generation Protocol

45 samples per model (5 seeds × 3 temperatures × 3 optimizers)

Seeds: محبت (love), دل (heart), شام (evening), یاد (memory), خوشی (happiness)
Temperatures: 0.7 (conservative), 1.0 (balanced), 1.3 (creative)

Metrics evaluated:
- Vocabulary diversity (unique/total words)
- Repetition rate (repeated bigrams)
- Average word length (characters)

---

## Running the Code

### 1. Install dependencies
pip install tensorflow numpy pandas scikit-learn datasets

### 2. Preprocess data
python preprocess.py

### 3. Train models (run sequentially or parallel)
python train_rnn.py
python train_lstm.py  
python train_transformer.py

### 4. Generate text samples
python generate_rnn.py
python generate_lstm.py
python generate_transformer.py

### 5. Analyze results
python analyze_results.py          # Comparison tables & plots
python analyze_generations.py      # Generation quality metrics

---

## Output Files

Results: *_results.pkl (test metrics: perplexity, accuracy, loss, time)
Models: *_model.keras (trained weights)
Plots: *_training.png (loss/accuracy curves)
Generations: *_generations.csv (45 samples per model)
Metrics: generation_metrics_*.csv (quantitative analysis)

---

## Evaluation Metrics

Primary: Perplexity (lower = better)
Secondary: Accuracy, training time
Qualitative: Fluency, coherence, poetic quality (manual evaluation)

---

## Notes

- Models use sparse_categorical_crossentropy (efficient for large vocab)
- N-gram approach eliminates need for causal masking
- Text is tokenized word-level (Keras Tokenizer)
- Urdu RTL direction handled naturally by tokenization order
- Test set (10%) held out for final evaluation only

---
