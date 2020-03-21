import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import tensorflow.keras as keras
import numpy as np
import os
import sys

glove = pd.read_csv("glove_6B_100d_top100k.csv")
glove.head()

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def find_nearest(embedding, word=None, n=5, distance=euclidean_distances):
    """
    For given embedding matrix and a given word, find the n nearest words in the embedding space

    input:
        embedding: DataFrame, look at `glove`
        word: string, must be in the index of embedding dataframe
        n: int, number of nearest words
        distance: fucntion, it should at least support the euclidean_distances and cosine_distances

    return:
        A series with word as index, distance as value, sorted from lower to high
    """
    """
    Write your code here
    """
    if word in embedding.columns:
        word_mat = embedding.loc[:, word].to_numpy().reshape(1, -1)

        words_mat = embedding.to_numpy()
        distances = np.zeros(words_mat.shape[1])
        for i in range(words_mat.shape[1]):
            distances[i] = distance(word_mat, words_mat[:, i].reshape(1, -1))

        index = distances.argsort()
        nearest = embedding.columns[index[:n+1]][1:]
        nearest_distance = distances[index[:n+1]][1:]

        return pd.Series(data=nearest_distance, index=nearest)

    else:
        print("word must be in embedding columns !")
        return None


print("Using euclidean_distances, the closest words to frog are:")
print(find_nearest(glove, 'lion'))
print("Using cosine_distances, the closest words to frog are:")
print(find_nearest(glove, 'lion', distance=cosine_distances))

print(find_nearest(glove, 'China'))
print(find_nearest(glove, 'China', distance=cosine_distances))

print(find_nearest(glove, '8'))
print(find_nearest(glove, '8', distance=cosine_distances))

print(find_nearest(glove, '6'))
print(find_nearest(glove, '6', distance=cosine_distances))

print(find_nearest(glove, 'USA'))
print(find_nearest(glove, 'USA', distance=cosine_distances))


def find_nearest_with_vector(embedding, vector=None, n=5, distance=euclidean_distances):
    """
    For given embedding matrix and a given vector, find the n nearest words in the embedding space

    input:
        embedding: DataFrame, look at `glove`
        vector: Series, looks like a coloumn vector of the embedding dataframe
        n: int, number of nearest words
        distance: fucntion, it should at least support the euclidean_distances and cosine_distances

    return:
        A series with word as index, distance as value, sorted from lower to high
    """
    """
    Write your code here
    """
    word_mat = vector.to_numpy().reshape(1, -1)
    words_mat = embedding.to_numpy()
    distances = np.zeros(words_mat.shape[1])
    for i in range(words_mat.shape[1]):
        distances[i] = distance(word_mat, words_mat[:, i].reshape(1, -1))

    index = distances.argsort()
    nearest = embedding.columns[index[:n]]
    nearest_distance = distances[index[:n]]

    return pd.Series(data=nearest_distance, index=nearest)


find_nearest_with_vector(glove, glove['king']-glove['male']+glove['female'])
find_nearest_with_vector(glove, glove['china']+glove['capital'])


from sklearn.decomposition import PCA


def plot_2D(X, labels):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = 0.1 + 0.8 * (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 8))
    for x, lab in zip(X, labels):
        plt.text(x[0], x[1], str(lab), fontdict={'size': 14})


def plot_words_embedding(embedding, words):
    X = PCA(n_components=2).fit_transform(embedding[words].transpose())
    plot_2D(X, words)


words = ['china', 'beijing', 'russia', 'moscow', 'poland', 'warsaw', 'japan', 'tokyo',
        'france', 'paris', 'germany', 'berlin', 'italy', 'rome', 'spain', 'madrid']

plot_words_embedding(glove, words)


from utils.hw3 import load_data
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


text = load_data("plot_summaries_tokenized.txt")

print("Number of summarys: ", len(text))
print("Number of words:", len([w for s in text for w in s]))
print("Vocabulary size:", len({w for s in text for w in s}))

wnl = WordNetLemmatizer()
stop_words = stopwords.words('english') + \
             ['~', '`', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '|', '/', '.', ',', ';', ':', "'", '"']
stop_words = [wnl.lemmatize(i) for i in stop_words]

for i in range(len(text)):
    text[i] = [wnl.lemmatize(j) for j in text[i] if j not in stop_words]

MIN_COUNT = 20


def create_encoder(text, min_count=20):
    """
    - Create a encoder which is a dictionary like {word: index}
    - To reduce the total number of vocabularies, you can remove
    the words that appear for less than min_count times in the entire
    corpus
    - Enfore {'_unknown_': 0}

    input:
        text: list of token list, e.g. [['i', 'am', 'fine'], ['another', 'summary'], ...]
    returns:
        tokenmap:  encoder dictionary
        tokenmap_reverse: reversed tokenmap {index: word} to faciliate inverse lookup
    """

    """
    Write your code here
    """
    word_counter = {}
    tokens = {'_unknown_'}

    for i in text:
        for j in i:
            if j not in tokens:
                if j in word_counter:
                    word_counter[j] += 1

                    if word_counter[j] >= min_count:
                        tokens.add(j)

                else:
                    word_counter[j] = 1

    tokenmap = {k: v for v, k in enumerate(tokens)}
    tokenmap_reverse = {v: k for v, k in enumerate(tokens)}

    return tokenmap, tokenmap_reverse


tokenmap, tokenmap_reverse = create_encoder(text, MIN_COUNT)
VOCAB_SIZE = len(tokenmap)
print("the reduced vocabulary size is:", VOCAB_SIZE)


# Encoder the text using the encoder you just created
def encode(text, tokenmap, default=0):
    return [[tokenmap.get(t, default) for t in s] for s in text]


text_encoded = encode(text, tokenmap)

from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
import random


def training_data_generator(text_encoded, window_size=4, negative_samples=1.0, batch_docs=50):
    """
    For given encoded text, return 3 np.array:
    words, contexts, labels
    Do not pair the w and its context cross different documents.

    input:
        text_encoded: list of list of int, each list of int is the numerical encoding of the doc
        window_size: int, define the context
        negative_samples: float, how much negative sampling you need, normally 1.0
        batch_docs: int, number of docs for which it generates one return

    return:
        words: list of int, the numerical encoding of the central words
        contexts: list of int, the numerical encoding of the context words
        labels: list of int, 1 or 0

    hint:
    1. You can use skipgrams method from keras
    2. For training purpose, words and contexts needs to be 2D array, with shape (N, 1),
       but labels is 1D array, with shape (N, )
    3. The output can be very big, you SHOULD using generator
    """

    """
    Write your code here
    """
    sampling_table = make_sampling_table(VOCAB_SIZE)
    loc = list(range(len(text_encoded)))
    random.shuffle(loc)

    for j in loc[:batch_docs]:
        couples, label = skipgrams(text_encoded[j], VOCAB_SIZE,
                                    window_size=window_size,
                                    sampling_table=sampling_table,
                                    negative_samples=negative_samples,
                                    shuffle=True)

        if len(couples) > 0:
            target, context_ = zip(*couples)
            target = np.array(target, dtype="int32")
            context_ = np.array(context_, dtype="int32")

            yield target.tolist(), context_.tolist(), label

        else:
            continue


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Reshape, dot, Dense


embedding_dim = 100

input_word = Input((1,))
input_context = Input((1,))

embedding_w = Embedding(VOCAB_SIZE, output_dim=embedding_dim, input_length=1, name='embedding_w')
embedding_c = Embedding(VOCAB_SIZE, output_dim=embedding_dim, input_length=1, name='embedding_c')

word = embedding_w(input_word)
word = Reshape((embedding_dim, 1))(word)

context = embedding_c(input_context)
context = Reshape((embedding_dim, 1))(context)

product = dot([word, context], axes=1)
product = Reshape((1,))(product)

output = Dense(1, activation='sigmoid')(product)

model = Model([input_word, input_context], output)
model.compile(loss="binary_crossentropy", optimizer="rmsprop")

model.summary()

epochs = 10
ntot = 0
for epoch in range(epochs):
    print("Epoch %d ======" % epoch)
    for words, contexts, labels in training_data_generator(text_encoded, batch_docs=50):
        loss = model.train_on_batch(x=[words, contexts], y=labels)
        ntot += len(words)
        print("Total trained pairs (M): %10.2f ; \t loss: %.4f" % (ntot/1e6, loss))


def embedding2df(embedding_layer, tokenmap_reverse):
    return pd.DataFrame(embedding_layer.get_weights()[0],
                        tokenmap_reverse.values()).drop("_unknown_", errors='ignore')


skip = embedding2df(model.layers[2], tokenmap_reverse)

print("Using euclidean_distances, the closest words to frog are:")
print(find_nearest(skip.T, 'lion'))
print("Using cosine_distances, the closest words to frog are:")
print(find_nearest(skip.T, 'lion', distance=cosine_distances))

