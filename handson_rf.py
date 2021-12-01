import pickle
import os.path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import accuracy_score


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

    random_state = 65

    mnist = open_mnist_or_download_if_missing()

    train_idx, test_idx = next(
        ShuffleSplit(n_splits=1, train_size=60000, random_state=random_state).split(mnist.data, mnist.target))
    train_idx, cv_idx = next(
        ShuffleSplit(n_splits=1, train_size=50000, random_state=random_state).split(mnist.data.iloc[train_idx],
                                                                                    mnist.target.iloc[train_idx]))

    X_train = mnist.data.iloc[train_idx]
    y_train = mnist.target.iloc[train_idx]

    X_cv = mnist.data.iloc[cv_idx]
    y_cv = mnist.target.iloc[cv_idx]

    X_test = mnist.data.iloc[test_idx]
    y_test = mnist.target.iloc[test_idx]

    # svc_clf = SVC(max_iter=100, probability=True, random_state=random_state)
    linear_svc_clf = LinearSVC(max_iter=100, random_state=random_state)
    rf_clf = RandomForestClassifier(random_state=random_state)
    lr_clf = LogisticRegression(random_state=random_state)
    extra_tree_clf = ExtraTreeClassifier(random_state=random_state)
    classifiers = [linear_svc_clf, rf_clf, lr_clf, extra_tree_clf]

    for clf in classifiers:
        print(f"Training {clf}")
        clf.fit(X_train, y_train)

    print(f"Classifiers:{classifiers}")
    print(f"Scores: {[clf.score(X_cv, y_cv) for clf in classifiers]}")

    named_estimators = [("random_forest", rf_clf), ("logistic_regression", lr_clf),
                        ("extra_tree", extra_tree_clf)]
    voting_clf = VotingClassifier(named_estimators)

    voting_clf.fit(X_train, y_train)

    print(f"hard voting classifier score: {voting_clf.score(X_cv, y_cv)}")
    print(voting_clf.estimators_)
    print(dir(voting_clf))
    print(f"Scores: {[clf.score(X_cv, y_cv) for clf in voting_clf.estimators_]}")

    voting_clf.voting = "soft"

    print(f"soft voting classifier score: {voting_clf.score(X_cv, y_cv)}")
    print(f"Scores: {[clf.score(X_cv, y_cv) for clf in voting_clf.estimators_]}")
