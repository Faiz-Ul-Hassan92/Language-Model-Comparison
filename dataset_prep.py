from datasets import load_dataset
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

dataset = load_dataset("ReySajju742/Urdu-Poetry-Dataset")

df = pd.DataFrame(dataset['train'])
print(f"Total poems: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

print(df['content'][0])


avg_words = 0

lines = []
poem_lengths = []

for content in df['content']:
    if content is not None:
        poem_lines = content.strip().split('\n')
        poem_lengths.append(len(poem_lines))
        for line in poem_lines:
            line = line.strip()
            if line:
                lines.append(line)
                avg_words += len(line)


plt.hist(poem_lengths, bins=30)
plt.title("Poem Length Distribution (Lines)")
plt.xlabel("Lines")
plt.ylabel("Count")
plt.show()






print(f"Total lines we have :{len(lines)}")

avg_size = len(lines)

avg_size = avg_size/len(df)
print(f"Avg Poem Size by lines: {avg_size}")


print(f"Average words per poem : {avg_words/len(df)}")




tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(lines)
vocab_size = len(tokenizer.word_index) + 1

print(f"Vocab Size with Keras Tokenizer: {vocab_size}")

print(f"Token Examples: {list(tokenizer.word_index.items())[:10]}")




line_token_lengths = [len(tokenizer.texts_to_sequences([line])[0]) for line in lines]

plt.hist(line_token_lengths, bins=40)
plt.title("Line Length Distribution (Token Count)")
plt.xlabel("Tokens")
plt.ylabel("Count")
plt.show()



#top 10 tokens

from collections import Counter

all_tokens = []
for line in lines:
    all_tokens.extend(tokenizer.texts_to_sequences([line])[0])

freq = Counter(all_tokens)
top_10 = freq.most_common(10)
top_30_to_40 = freq.most_common()[30:40]
top_words = [(tokenizer.index_word[t], c) for t, c in top_10]
print("\nTop 10 words (word, frequency):")
print(top_words)

top_30_to_40_words = [(tokenizer.index_word[t], c) for t, c in top_30_to_40]
print("\nTop 30 to 40 ranked words (word, frequency):")
print(top_30_to_40_words)





sequences = []
seq_lengths = []

for line in lines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)
        seq_lengths.append(len(n_gram_sequence))


plt.hist(seq_lengths, bins=40)
plt.title("N-gram Sequence Length Distribution")
plt.xlabel("Length")
plt.ylabel("Count")
plt.show()



print(f"Total sequences for training {len(sequences)}")

for i in range(10,15):
    print(sequences[i])


max_sequence_len = max([len(seq) for seq in sequences])


print(max_sequence_len)
max_sequence_len = 20 #baselines
print(f"Fixing max sequence length to {max_sequence_len}")


sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
#i just learnt this is just truncating and not removing longer sentences


print(f"\nSequences shape after padding: {sequences.shape}")
print(f"Sample padded sequence:\n{sequences[100]}")

X = sequences[:, :-1]
y = sequences[:, -1] #separated the token to be predicted

print(f"\nX shape (input): {X.shape}")
print(f"y shape (target): {y.shape}")


plt.hist(y, bins=50)
plt.title("Target Token Distribution")
plt.xlabel("Token ID")
plt.ylabel("Frequency")
plt.show()

from collections import Counter

freq_y = Counter(y)
top_5_y = freq_y.most_common(5)

print("Top 5 target tokens (token_id, frequency):")
print(top_5_y)

print("\nTop 5 target words (word, frequency):")
for token_id, count in top_5_y:
    word = tokenizer.index_word.get(token_id, "<UNK>")
    print(word, count)


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
