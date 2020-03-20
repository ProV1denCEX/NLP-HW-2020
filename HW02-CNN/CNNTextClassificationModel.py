import os
import sys
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from utils.hw2 import load_data, save_prediction, read_vocab
from utils.general import sigmoid, tanh, show_keras_model


wnl = WordNetLemmatizer()
stop_words = stopwords.words('english') + \
             ['~', '`', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '|', '/', '.', ',', ';', ':', "'", '"']
stop_words = [wnl.lemmatize(i) for i in stop_words]

tanh_v = np.vectorize(tanh)
sigmoid_v = np.vectorize(sigmoid)


class CNNTextClassificationModel:
    def __init__(self, vocab, window_size=2, F=100, alpha=0.1):
        """
        F: number of filters
        alpha: back propagatoin learning rate
        """
        self.vocab = vocab
        self.window_size = window_size
        self.F = F
        self.alpha = alpha

        # U and w are the weights of the hidden layer, see Fig 1 in the pdf file
        # U is the 1D convolutional layer with shape: voc_size * num_filter * window_size
        self.U = np.random.normal(loc=0, scale=0.01, size=(len(vocab), F, window_size))
        # w is the weights of the activation layer (after max pooling)
        self.w = np.random.normal(loc=0, scale=0.01, size=(F + 1))

    def pipeline(self, X):
        """
        Data processing pipeline to:
        1. Tokenize, Normalize the raw input
        2. Translate raw data input into numerical encoded vectors

        :param X: raw data input
        :return: list of lists

        For example:
        X = [["Apples orange banana"]
         ["orange apple bananas"]]
        returns:
        [[0, 1, 2],
         [1, 0, 2]]
        """

        """
        Implement your code here
        """
        X2 = []

        for i in X:
            words = word_tokenize(i)
            words = [wnl.lemmatize(j) for j in words if j not in stop_words]
            encoded = [vocab.get(j, 10000) for j in words]

            if len(encoded) >= self.window_size:
                X2.append(encoded)

        return X2

    @staticmethod
    def accuracy(probs, labels):
        assert len(probs) == len(labels), "Wrong input!!"
        a = np.array(probs)
        b = np.array(labels)

        return 1.0 * (a == b).sum() / len(b)


    def train(self, X_train, y_train, X_dev, y_dev, nEpoch=50):
        """
        Function to fit the model
        :param X_train, X_dev: raw data input
        :param y_train, y_dev: label
        :nEpoch: number of training epoches
        """
        X_train = self.pipeline(X_train)
        X_dev = self.pipeline(X_dev)

        for epoch in range(nEpoch):
            self.fit(X_train, y_train)

            accuracy_train = self.accuracy(self.predict(X_train), y_train)
            accuracy_dev = self.accuracy(self.predict(X_dev), y_dev)

            print("Epoch: {}\tTrain accuracy: {:.3f}\tDev accuracy: {:.3f}"
                  .format(epoch, accuracy_train, accuracy_dev))


    def fit(self, X, y):
        """
        :param X: numerical encoded input
        """
        for (data, label) in zip(X, y):
            self.backward(data, label)

        return self


    def predict(self, X):
        """
        :param X: numerical encoded input
        """
        result = []
        for data in X:
            if self.forward(data)["prob"] > 0.5:
                result.append(1)
            else:
                result.append(0)

        return result


    def forward(self, word_indices):
        """
        :param word_indices: a list of numerically ecoded words
        :return: a result dictionary containing 3 items -
        result['prob']: \hat y in Fig 1.
        result['h']: the hidden layer output after max pooling, h = [h1, ..., hf]
        result['hid']: argmax of F filters, e.g. j of x_j
        e.g. for the ith filter u_i, tanh(word[hid[j], hid[j] + width]*u_i) = h_i
        """

        assert len(word_indices) >= self.window_size, "Input length cannot be shorter than the window size"

        h = np.zeros(self.F + 1, dtype=float)
        h[0] = 1
        hid = np.zeros(self.F, dtype=int)
        prob = 0.0

        # layer 1. compute h and hid
        # loop through the input data of word indices and
        # keep track of the max filtered value h_i and its position index x_j
        # h_i = max(tanh(weighted sum of all words in a given window)) over all windows for u_i
        """
        Implement your code here
        """
        num_x = len(word_indices) - self.window_size + 1
        x = []
        for i in range(num_x):
            x.append([word_indices[i], word_indices[i + 1]])

        for i in range(self.F):
            ux = np.zeros(num_x)
            for j in range(len(x)):
                tmp = tanh_v(self.U[x[j], i, :].sum(axis=0))

                ux[j] = max(tmp)

            h[i+1] = ux.max()
            hid[i] = ux.argmax()

        # layer 2. compute probability
        # once h and hid are computed, compute the probabiliy by sigmoid(h^TV)
        """
        Implement your code here
        """
        prob = sigmoid(np.dot(self.w.transpose(), h))

        # return result
        return {"prob": prob, "h": h, "hid": hid}

    def backward(self, word_indices, label):
        """
        Update the U, w using backward propagation

        :param word_indices: a list of numerically ecoded words
        :param label: int 0 or 1
        :return: None

        update weight matrix/vector U and V based on the loss function
        """

        pred = self.forward(word_indices)
        prob = pred["prob"]
        h = pred["h"]
        hid = pred["hid"]

        # update U and w here
        # to update V: w_new = w_current + d(loss_function)/d(w)*alpha
        # to update U: U_new = U_current + d(loss_function)/d(U)*alpha
        # Hint: use Q6 in the first part of your homework
        """
        Implement your code here
        """
        grad = self.calc_gradients_w(pred, label)
        self.w += grad * self.alpha
        grad = self.calc_gradients_U(pred, word_indices)
        self.U += grad * self.alpha

    def calc_gradients_U(self, pred, word_indices):
        x = [[word_indices[i], word_indices[i + 1]] for i in pred['hid']]
        x = np.array(x)
        grads = np.zeros((len(vocab), self.F, self.window_size))
        for i in range(len(x)):
            grads[x[i], i, :] = 1 - np.power(pred['h'][i+1], 2)
        return grads

    def calc_gradients_w(self, pred, y):
        return (y - pred['prob']) * pred['h']


if __name__ == '__main__':
    """
    This cell shows you how the model will be used, you have to finish the cell below before you
    can run this cell. 

    Once the implementation is done, you should hype tune the parameters to find the best config

    Note I only selected 2000 data points to speed up debugging, you should use all the data to train the 
    final model
    """

    from sklearn.model_selection import train_test_split

    data = load_data("train.txt")
    vocab = read_vocab("vocab.txt")
    X, y = data.text, data.target
    X_train, X_dev, y_train, y_dev = train_test_split(X[:2000], y[:2000], test_size=0.3)
    cls = CNNTextClassificationModel(vocab)
    cls.train(X_train, y_train, X_dev, y_dev, nEpoch=10)

    data = load_data("test.txt")

    X, y = data.text, data.target

    y_predict = cls.predict(X)

    save_prediction(y_predict)
