import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse

# add utils folder to path
p = os.path.dirname(os.getcwd())
if p not in sys.path:
    sys.path = [p] + sys.path

from utils.hw5 import load_data, save_prediction, ignore_class_accuracy, whole_sentence_accuracy
from utils.general import show_keras_model

tags = list(pd.read_csv('tags.csv', index_col=0).tag_encode.keys())

train, train_label = load_data("train.txt")
train, dev, train_label, dev_label = train_test_split(train, train_label)
test, _ = load_data("test.txt")

print("Training set: %d" % len(train))
print("Dev set: %d" % len(dev))
print("Testing set: %d" % len(test))

from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Dropout
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.utils import to_categorical


class POS_LSTMM:
    """
    To help you focus on the LSTM model, I have made most part of the code ready, make sure you
    read all the parts to understand how the code works. You only need to modify the prepare method
    to add the RNN model.
    """

    def __init__(self, tag_vocab=tags, max_sent_len=40,
                 voc_min_freq=5, **kwargs):
        """
        input:
            tag_vocab: tag dictionary, you will less likely need to change this
            voc_min_freq: use this to truncate low frequency vocabulary
            max_sent_len: truncate/pad all sentences to this length

            kwargs: Use as needed to pass extra parameters
        """
        self.vocab = []
        self.reverse_vocab = {}
        self.tag_vocab = tag_vocab
        self.reverse_tag_vocab = {k: v for v, k in enumerate(tag_vocab)}
        self._voc_min_freq = voc_min_freq
        self._max_sent_len = max_sent_len

        """
        Feel free to add code here as you need
        """

    def collect_vocab(self, X):
        """
        Create vocabulary from all input data
        input:
            X: list of sentences
        """
        vocab = Counter([t for s in X for t in s])
        vocab = {k: v for k, v in vocab.items() if v > self._voc_min_freq}
        vocab = ["<PAD>", "<UNK>"] + sorted(vocab, key=lambda x: vocab[x], reverse=True)
        reverse_vocab = {k: v for v, k in enumerate(vocab)}

        return vocab, reverse_vocab

    def transform_X(self, X):
        """
        Translate input raw data X into trainable numerical data
        input:
            X: list of sentences
        """
        X_out = []

        default = self.reverse_vocab['<UNK>']
        for sent in X:
            X_out.append([self.reverse_vocab.get(t, default) for t in sent])

        X_out = pad_sequences(sequences=X_out, maxlen=self._max_sent_len,
                              padding='post', truncating='post',
                              value=self.reverse_vocab['<PAD>'])

        return X_out

    def transform_Y(self, Y):
        """
        Translate input raw data Y into trainable numerical data
        input:
            y: list of list of tags
        """
        Y_out = []

        for labs in Y:
            Y_out.append([self.reverse_tag_vocab[lab] for lab in labs])

        Y_out = pad_sequences(sequences=Y_out, maxlen=self._max_sent_len,
                              padding='post', truncating='post',
                              value=self.reverse_tag_vocab['<PAD>'])

        return Y_out

    def prepare(self, X, Y):
        """
        input:
            X: list of sentences
            y: list of list of tags
        """
        self.vocab, self.reverse_vocab = self.collect_vocab(X)
        X, Y = self.transform_X(X), self.transform_Y(Y)

        embedding_dim = 100
        lstm_node = 128

        """
        Write your own model here
        Hints:
            - Rember to use embedding layer at the beginning
            - Use Bidrectional LSTM to take advantage of both direction history  
        """
        model = Sequential()
        model.add(InputLayer(input_shape=(self._max_sent_len,)))
        model.add(Embedding(len(self.vocab), embedding_dim))
        model.add(Bidirectional(LSTM(lstm_node, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(len(self.tag_vocab))))
        model.add(Dropout(0.2))
        model.add(Activation('softmax'))

        """
        You can read the source code to understand how ignore_class_accuracy works.
        The reason of using this customized metric is because we have padded the training 
        data with lots of '<PAD>' tag. It's easy and useless to predict this tag, we need 
        to ignore this tag when calculate the accuracy.
        """
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.001),
                      metrics=['accuracy',
                               ignore_class_accuracy(self.reverse_tag_vocab['<PAD>']),
                               whole_sentence_accuracy(self.reverse_tag_vocab['<PAD>'])])

        self.model = model

        return self

    def fit(self, X, Y, batch_size=128, epochs=10):
        X, Y = self.transform_X(X), self.transform_Y(Y)
        self.model.fit(X, to_categorical(Y, num_classes=len(self.tag_vocab)),
                       batch_size=batch_size,
                       epochs=epochs, validation_split=0.2)

        return self

    def predict(self, X):
        results = []
        X_new = self.transform_X(X)
        Y_pred = self.model.predict_classes(X_new)

        for i, y in enumerate(Y_pred):
            results.append(
                [self.tag_vocab[y[j]] for j in range(min(len(X[i]), len(X_new[i])))]
            )

        return results

    def evaluate(self, X, Y, batch_size=128):
        X, Y = self.transform_X(X), self.transform_Y(Y)
        results = self.model.evaluate(X, to_categorical(Y, num_classes=len(self.tag_vocab)), batch_size=batch_size)

        return results


lstm = POS_LSTMM().prepare(train, train_label)
lstm.model.summary()
# show_keras_model(lstm.model)

lstm = POS_LSTMM().prepare(train, train_label)
lstm.fit(train, train_label)
results = lstm.evaluate(dev, dev_label, batch_size=128)
print('test loss, test acc:', results)

prediction = lstm.predict(test)
save_prediction(prediction)

