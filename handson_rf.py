import pickle
import os.path

from sklearn.datasets import fetch_openml


def open_mnist_or_download_if_missing():
    if os.path.exists("data/mnist/mnist.pickle"):
        print("Using local mnist")
        with open("data/mnist/mnist.pickle", mode="rb") as fp:
            return pickle.load(fp)
    else:
        print("Downloading mnist")
        mnist = fetch_openml("mnist_784", version=1)
        with open("data/mnist/mnist.pickle", mode="wb") as fp:
            pickle.dump(mnist, fp)
        return mnist


if __name__ == "__main__":
    mnist = open_mnist_or_download_if_missing()

    print(mnist.keys())
