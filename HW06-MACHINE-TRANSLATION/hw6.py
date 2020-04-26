from tensorflow.keras.layers import (Input, LSTM, Dense, Bidirectional, Embedding,
                          TimeDistributed, Concatenate)
import numpy as np


encoder_input_len = 11
decoder_input_len = 10
latent_dim = 100

"""
Now are you ready for the real challenge? You can use the ita.txt file as training data. 
But feel free to download different language from http://www.manythings.org/anki/. If you
happen to speak French or Japanese, it's time to show off!

1. Implement a Bidrectional LSTM Encoder-Decoder model, or other viable models to translate 
   the language dataset you choose.

2. Write the function to calculate the BLEU score of your model
"""
import pandas as pd
import re
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from keras.initializers import Constant

lines= pd.read_csv('spa.txt', names=['eng', 'spa', 'source'], sep='\t')

# Lowercase all characters
lines.eng=lines.eng.apply(lambda x: x.lower())
lines.spa=lines.spa.apply(lambda x: x.lower())

# Remove quotes
lines.eng=lines.eng.apply(lambda x: re.sub("'", '', x))
lines.spa=lines.spa.apply(lambda x: re.sub("'", '', x))
exclude = set(string.punctuation) # Set of all special characters

# Remove all the special characters
lines.eng=lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.spa=lines.spa.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# Remove all numbers from text
remove_digits = str.maketrans('', '', string.digits)
lines.eng=lines.eng.apply(lambda x: x.translate(remove_digits))
lines.spa = lines.spa.apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remove extra spaces
lines.eng=lines.eng.apply(lambda x: x.strip())
lines.spa=lines.spa.apply(lambda x: x.strip())
lines.eng=lines.eng.apply(lambda x: re.sub(" +", " ", x))
lines.spa=lines.spa.apply(lambda x: re.sub(" +", " ", x))

# Add start and end tokens to target sequences
start_token = '<START> '
end_token = '<END>'
lines.spa = lines.spa.apply(lambda x : ''.join([start_token, x, end_token]))

# Vocabulary of English
all_eng_words=set()
for eng in lines.eng:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

# Vocabulary of French
all_spa_words=set()
for spa in lines.spa:
    for word in spa.split():
        if word not in all_spa_words:
            all_spa_words.add(word)

# Max Length of source sequence
lenght_list=[]
for l in lines.eng:
    lenght_list.append(len(l.split(' ')))
max_length_src = np.max(lenght_list)
max_length_src

# Max Length of target sequence
lenght_list=[]
for l in lines.spa:
    lenght_list.append(len(l.split(' ')))
max_length_tar = np.max(lenght_list)

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_spa_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_spa_words)

# For zero padding
num_decoder_tokens += 1
num_encoder_tokens += 1

input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])
reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

lines = shuffle(lines)

# Train - Test Split
X, y = lines.eng, lines.spa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


def generate_batch(X=X_train, y=y_train, batch_size=128):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar), dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens), dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j: j + batch_size], y[j: j + batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word]  # encoder input seq
                for t, word in enumerate(target_text.split()):
                    if t < len(target_text.split()) - 1:
                        decoder_input_data[i, t] = target_token_index[word]  # decoder input seq
                    if t > 0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START token
                        # Offset by one timestep
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.

            yield [encoder_input_data, decoder_input_data], decoder_target_data, [None]


glove = pd.read_csv("glove_6B_100d_top100k.csv")

embedding_matrix = np.zeros((num_encoder_tokens, 100))
for word, i in input_token_index.items():
    if word in glove.columns:
        embedding_matrix[i] = glove.loc[:, word].to_numpy()


encoder_inputs = Input(shape=(None,), name="Encoder_Input")
enc_emb =  Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)

# For encoder, we can see the entire sentence at once, so we can use Bidirectional LSTM
encoder_lstm = Bidirectional(LSTM(latent_dim, return_state=True, name="Encoder_LSTM"))
# Bidrectional LSTM has 4 states instead of 2, we concatenate them to be comparable
# with the decoder LSTM
_, forward_h, forward_c, backward_h, backward_c = encoder_lstm(enc_emb)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

# Set up the decoder, using `encoder_states` as initial state
encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None,), name="Decoder_Input")
dec_emb = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)

decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True, name="Decoder_LSTM")
decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = TimeDistributed(decoder_dense)(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 128
epochs = 5

model.fit(generate_batch(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch = train_samples // batch_size,
                    epochs=epochs,
                    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                    validation_steps = val_samples // batch_size)

# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index[start_token]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == end_token or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


val_gen = generate_batch(X_test, y_test, batch_size = 1)
(input_seq, actual_output), _, _ = next(val_gen)
decoded_sentence = decode_sequence(input_seq)

import nltk

hypothesis = decoded_sentence
reference = actual_output
#there may be several references
BLEU_score = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)

print(BLEU_score)
