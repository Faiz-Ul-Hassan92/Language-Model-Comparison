from datasets import load_dataset
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

np.random.seed(42)

dataset = load_dataset("ReySajju742/Urdu-Poetry-Dataset")

df = pd.DataFrame(dataset['train'])
print(f"Total poems: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

print(df['content'][0])



lines = []
for content in df['content']:
    if content is not None:
        poem_lines = content.strip().split('\n')
        for line in poem_lines:
            line = line.strip()
            if line:
                lines.append(line)


print(f"Total lines we have :{len(lines)}")

avg_size = len(lines)

avg_size = avg_size/len(df)
print(f"Avg Poem Size by lines: {avg_size}")




tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(lines)
vocab_size = len(tokenizer.word_index) + 1

print(f"Vocab Size with Keras Tokenizer: {vocab_size}")

print(f"Token Examples: {list(tokenizer.word_index.items())[:10]}")





sequences = []
for line in lines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)



print(f"Total sequences for training {len(sequences)}")

for i in range(10,15):
    print(sequences[i])


max_sequence_len = max([len(seq) for seq in sequences])

print(max_sequence_len)


sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

print(f"\nSequences shape after padding: {sequences.shape}")
print(f"First padded sequence:\n{sequences[0]}")

X = sequences[:, :-1]
y = sequences[:, -1] #separated the token to be predicted

print(f"\nX shape (input): {X.shape}")
print(f"y shape (target): {y.shape}")


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)




print(X_train.shape)



np.save('datasetFormatted/X_train.npy', X_train)
np.save('datasetFormatted/X_val.npy', X_val)
np.save('datasetFormatted/X_test.npy', X_test)
np.save('datasetFormatted/y_train.npy', y_train)
np.save('datasetFormatted/y_val.npy', y_val)
np.save('datasetFormatted/y_test.npy', y_test)

import pickle
with open('tokenizer/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('tokenizer/config.pkl', 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'max_sequence_len': max_sequence_len}, f)
