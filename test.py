#!/usr/bin/env python
import gzip
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

train_x, train_y = train_set

for i in range(len(train_x)):
    plt.imshow(train_x[i].reshape((28, 28)), cmap=cm.Greys_r)
    print(train_y[i])
    plt.show()