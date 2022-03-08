import os
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit


def open_mnist_or_download_if_missing():
    if os.path.exists("data/mnist/mnist.pickle"):
        print("Using local mnist")
        with open("data/mnist/mnist.pickle", mode="rb") as fp:
            mnist = pickle.load(fp)
    else:
        print("Downloading mnist")
        mnist = fetch_openml("mnist_784", version=1)
        with open("data/mnist/mnist.pickle", mode="wb") as fp:
            pickle.dump(mnist, fp)
    mnist.target = mnist.target.astype(np.uint8)
    return mnist


def shuffle_split(X, y, train_size, random_state=None):
    if isinstance(X, pd.DataFrame):
        X = X.iloc
        y = y.iloc
    for train_idx, test_idx in StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                                      random_state=random_state).split(X, y):
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    mnist = open_mnist_or_download_if_missing()
    X_train, y_train, X_test, y_test = shuffle_split(mnist["data"], mnist["target"], 60000, 33)
    print(X_train)