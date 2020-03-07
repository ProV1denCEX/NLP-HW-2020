import numpy as np
import pandas as pd
import sklearn as sk
import nltk
import scipy.sparse as sparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class PorterTokenizer:
    def __init__(self):
        self.porter = nltk.PorterStemmer()

    def __call__(self, doc):
        return [self.porter.stem(t) for t in nltk.word_tokenize(doc)]


stop_words = nltk.corpus.stopwords.words('english') + \
             ['~', '`', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '|', '/', '.', ',', ';', ':', "'", '"']
stop_words = sum([PorterTokenizer()(i) for i in stop_words], [])


def dumb_featurize(text, vocab=None):
    """
    due to the refactored SentimentClassifier.featurize,
    there are some changes in this func
    :param text: list of str, raw data
    :return: vectorized data, feature_vocab
    """
    tmp = []
    for i in text:
        feats = set()
        words = i.split(" ")

        for word in words:
            if word == "love" or word == "like" or word == "best":
                feats.add('contains_positive_word')

            elif word == "hate" or word == "dislike" or word == "worst" or word == "awful":
                feats.add('contains_negative_word')

            else:
                feats.add('__UNKNOWN__')

        tmp.append(' '.join(feats))

    if vocab is None:
        vectorizer = CountVectorizer(vocabulary=['__UNKNOWN__', 'contains_negative_word', 'contains_positive_word'])

    else:
        vectorizer = CountVectorizer(vocabulary=vocab)

    vec = vectorizer.fit_transform(tmp)
    features = vectorizer.get_feature_names()

    return vec, features


def bag_of_words(text, vocab=None):
    tokenizer = PorterTokenizer()

    if vocab is None:
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words, tokenizer=tokenizer)

    else:
        vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), stop_words=stop_words, tokenizer=tokenizer)

    counts = vectorizer.fit_transform(text)
    features = vectorizer.get_feature_names()

    # handle unknown data
    if vocab is not None:
        counts = counts.tolil()
        unknown_idx = features.index('__UNKNOWN__')

        for i in range(len(text)):
            counts[i, unknown_idx] = len(set(tokenizer(text[i])) - set(features)) - 1

    else:
        features.append('__UNKNOWN__')
        counts = sparse.hstack([counts, sparse.csr_matrix((len(text), 1))])

    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(counts)

    return tf_idf, features


def pos_weight_featurize(text, vocab=None):
    """
    main idea: people tend to show their attitude in most recent words in a comment.
                The words closer to the start should have more weights,
                only in 1 gram now
    :param text:
    :return:
    """
    tokenizer = PorterTokenizer()

    if vocab is None:
        vectorizer = CountVectorizer(stop_words=stop_words, tokenizer=tokenizer)

    else:
        vectorizer = CountVectorizer(vocabulary=vocab, stop_words=stop_words, tokenizer=tokenizer)

    counts = vectorizer.fit_transform(text)
    features = vectorizer.get_feature_names()

    # handle unknown data
    if vocab is not None:
        counts = counts.tolil()
        unknown_idx = features.index('__UNKNOWN__')
        features_pos = {feat: i for i, feat in enumerate(features)}

        for i in range(len(text)):
            tmp = tokenizer(text[i])
            counts[i, unknown_idx] = len(set(tmp) - set(features)) - 1

            weights = {feat: (len(tmp) - i + 1) / len(tmp) for i, feat in enumerate(tmp)}
            unknown_weight = 0
            for k, v in weights.items():
                if k in features:
                    pos = features_pos[k]
                    counts[i, pos] = counts[i, pos] * v

                else:
                    unknown_weight += v

            counts[i, unknown_idx] = counts[i, unknown_idx] * unknown_weight

    else:
        features.append('__UNKNOWN__')
        counts = sparse.hstack([counts, sparse.csr_matrix((len(text), 1))])

    return counts, features


def bag_of_words_with_pos(text, vocab):
    counts, features = pos_weight_featurize(text, vocab)

    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(counts)

    return tf_idf, features


class SentimentClassifier(object):
    def __init__(self, feature_method=dumb_featurize, min_feature_ct=1, L2_reg=1.0):
        """
        :param feature_method: featurize function
        :param min_feature_count: int, ignore the features that appear less than this number to avoid overfitting
        """
        self.feature_vocab = None
        self.feature_method = feature_method
        self.min_feature_ct = min_feature_ct
        self.L2_reg = L2_reg

    def featurize(self, X):
        """
        # Featurize input text

        :param X: list of texts
        :return: list of featurized vectors
        """
        return self.feature_method(X, self.feature_vocab)

    def pipeline(self, X, training=False):
        """
        Data processing pipeline to translate raw data input into sparse vectors
        :param X: featurized input
        :return X2: 2d sparse vectors

        Implement the pipeline method that translate the dictionary like feature vectors into
        homogeneous numerical vectors, for example:
        [{"fea1": 1, "fea2": 2},
         {"fea2": 2, "fea3": 3}]
         -->
         [[1, 2, 0],
          [0, 2, 3]]

        Hints:
        1. How can you know the length of the feature vector?
        2. When should you use sparse matrix?
        3. Have you treated non-seen features properly?
        4. Should you treat training and testing data differently?
        """
        # Have to build feature_vocab during training
        vec, feature_vocab = self.featurize(X)
        if training:
            # Get all words and # them
            self.feature_vocab = {feats: i for i, feats in enumerate(feature_vocab)}

        return vec

    def fit(self, X, y):
        X = self.pipeline(X, training=True)

        D, F = X.shape
        self.model = LogisticRegression(C=self.L2_reg)
        self.model.fit(X, y)

        return self

    def predict(self, X):
        X = self.pipeline(X)
        return self.model.predict(X)

    def score(self, X, y):
        X = self.pipeline(X)
        return self.model.score(X, y)

    # Write learned parameters to file
    def save_weights(self, filename='weights.csv'):
        weights = [["__intercept__", self.model.intercept_[0]]]
        for feat, idx in self.feature_vocab.items():
            weights.append([feat, self.model.coef_[0][idx]])

        weights = pd.DataFrame(weights)
        weights.to_csv(filename, header=False, index=False)

        return weights


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from utils.hw1 import load_data, save_prediction

    data = load_data("train.txt")
    X, y = data.text, data.target
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3)
    cls = SentimentClassifier(feature_method=bag_of_words)
    cls = cls.fit(X_train, y_train)

    print("Training set accuracy: ", cls.score(X_train, y_train))
    print("Dev set accuracy: ", cls.score(X_dev, y_dev))

    weights = cls.save_weights()

    X_test = load_data("test.txt").text
    save_prediction(cls.predict(X_test))