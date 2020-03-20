import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from utils.hw2 import load_data, save_prediction, read_vocab
from utils.general import sigmoid, tanh, show_keras_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, GlobalMaxPooling1D

from sklearn.model_selection import train_test_split

data = load_data("train.txt")

X, y = data.text, data.target
X_train, X_dev, y_train, y_dev = train_test_split(X[:2000], y[:2000], test_size=0.3)


MAX_LENGTH = 100
wnet = WordNetLemmatizer()
vocab = read_vocab("vocab.txt")

unknown = vocab['__unknown__']
X_train2 = [[vocab.get(wnet.lemmatize(w), unknown) for w in word_tokenize(sent)] for sent in X_train]
X_dev2 = [[vocab.get(wnet.lemmatize(w), unknown) for w in word_tokenize(sent)] for sent in X_dev]


def trim_X(X, max_length=100, default=vocab['.']):
    for i in range(len(X)):
        if len(X[i]) > max_length:
            X[i] = X[i][:max_length]
        elif len(X[i]) < max_length:
            X[i] = X[i] + [default] * (max_length - len(X[i]))
    return np.array(X)


X_train2 = trim_X(X_train2, MAX_LENGTH)
X_dev2 = trim_X(X_dev2, MAX_LENGTH)

model = Sequential()
model.add(Embedding(input_dim=len(vocab), input_length=MAX_LENGTH, output_dim=1024, name="Embedding-1"))
model.add(Conv1D(filters=100, kernel_size=2, activation="tanh", name="Conv1D-1"))
model.add(GlobalMaxPooling1D(name="MaxPooling1D-1"))
model.add(Dense(1, activation="sigmoid", name="Dense-1"))

print(model.summary())
# show_keras_model(model)

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(X_train2, y_train, epochs=10, validation_data=[X_dev2, y_dev])

data = load_data("test.txt")

X, y = data.text, data.target
X = [[vocab.get(wnet.lemmatize(w), unknown) for w in word_tokenize(sent)] for sent in X]

X = trim_X(X, MAX_LENGTH)
y_predict = model.predict(X)

save_prediction(y_predict)
