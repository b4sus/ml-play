import re

import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
import pandas as pd
from sklearn.base import TransformerMixin, clone
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score, plot_precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TextPreprocessor(TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, X):
        processed = []
        for x in X:
            processed.append(" ".join(self.preprocess_tweet_text(x)))
        return processed


    def preprocess_tweet_text(self, tweet_text):
        tweet_text = tweet_text.lower()
        tweet_text = re.sub("<[^<>]+>", " ", tweet_text)
        tweet_text = re.sub("(http|https)://[^\\s]*", "httpaddr", tweet_text)
        tweet_text = re.sub("[^\\s]+@[^\\s]+", "emailaddr", tweet_text)
        tweet_text = re.sub("[\\d]+", "number", tweet_text)
        tweet_text = re.sub("[$]+", "dollar", tweet_text)
        words = []
        for word in tweet_text.split():
            word_chars_only = re.sub("[^a-z]", "", word)
            if word_chars_only != "":
                words.append(word_chars_only)
        # lemmatizer = WordNetLemmatizer()
        # return [lemmatizer.lemmatize(word) for word in words]
        stemmer = EnglishStemmer()
        return [stemmer.stem(word) for word in words]


def svm_grid_search(X, y):
    param_grid = {"C": [0.1, 0.3, 1, 3, 10], "kernel": ["linear", "poly", "rbf"]}
    return find_best_estimator(GridSearchCV(SVC(), param_grid, cv=3), X, y)


def svm_rbf_grid_search(X, y):
    param_grid = {"C": [2.5, 2.8, 3, 3.2, 3.5], "gamma": [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]}
    # best estimator is SVC(C=2.8, gamma=0.05) with score 0.8052545155993432
    return find_best_estimator(GridSearchCV(SVC(kernel="rbf"), param_grid, cv=3), X, y)


def logistic_regression_grid_search(X, y, count_vectorizer_feature_names):
    # poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    # X = poly_features.fit_transform(X)
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), {"C": [0.1, 0.3, 0.7, 1, 3, 7, 10], "solver": ["newton-cg", "liblinear", "lbfgs"]}, cv=3)
    clf = find_best_estimator(grid_search, X, y)
    sorted_coef_inidces = np.argsort(clf.coef_).reshape((-1))
    print_most_significant_words(count_vectorizer_feature_names, sorted_coef_inidces[-1:-32:-1], sorted_coef_inidces[:20])
    return clf


def find_best_estimator(grid_search, X, y):
    print(f"Running grid search:\n{grid_search}")
    grid_search.fit(X, y)
    print(f"best estimator is {grid_search.best_estimator_} with score {grid_search.best_score_}")
    return grid_search.best_estimator_


def print_most_significant_words(count_vectorizer_feature_names, top_indices, flop_indices):
    print(f"top indices {top_indices}")
    print(f"flop indices {flop_indices}")
    print(f"Top words:{[count_vectorizer_feature_names[idx] for idx in top_indices if idx < len(count_vectorizer_feature_names)]}")
    print(f"Flop words:{[count_vectorizer_feature_names[idx] for idx in flop_indices if idx < len(count_vectorizer_feature_names)]}")


def knn_grid_search(X, y):
    param_grid = {"n_neighbors": [1, 2, 3, 4, 5, 10, 15, 20], "weights": ["uniform", "distance"]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
    return find_best_estimator(grid_search, X, y)


def nn_grid_search(X, y):
    param_grid = {"hidden_layer_sizes": [(5, 5, 5), (3, 5, 2), (10, 10)], "activation": ["logistic", "relu"]}
    # best estimator is MLPClassifier(activation='logistic', hidden_layer_sizes=(3, 5, 2),
    #               max_iter=1000) with score 0.7712643678160919
    return find_best_estimator(GridSearchCV(MLPClassifier(max_iter=1000), param_grid, cv=3), X, y)


def predict_test(clf, transformer, threshold=None):
    tweets = pd.read_csv("data/disaster_tweets/test.csv")
    X = transformer.transform(tweets)
    if threshold is None:
        predictions = clf.predict(X)
    else:
        predictions = (clf.decision_function(X) > threshold).astype(np.int32)
    submission = pd.concat({"id": tweets["id"], "target": pd.Series(predictions)}, axis=1)
    submission.set_index("id")
    submission.to_csv("data/disaster_tweets/submission.csv", index=False)


def analyse_errors(clf, X, y):
    y_pred = clf.predict(X)
    conf_mx = confusion_matrix(y, y_pred)
    print(conf_mx)
    print(f"accuracy {accuracy_score(y, y_pred)}, precision {precision_score(y, y_pred)}, recall {recall_score(y, y_pred)}, f1 {f1_score(y, y_pred)}")
    precisions, recall, thresholds = precision_recall_curve(y, y_pred)


def analyse_errors_cross_val(clf, X, y):
    y_pred = cross_val_predict(clf, X, y, cv=3)
    conf_mx = confusion_matrix(y, y_pred)
    print(conf_mx)
    print(
        f"accuracy {accuracy_score(y, y_pred)}, precision {precision_score(y, y_pred)}, recall {recall_score(y, y_pred)}, f1 {f1_score(y, y_pred)}")
    y_scores = cross_val_predict(clf, X, y, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    plt.plot(thresholds, precisions[:-1], label="precision")
    plt.plot(thresholds, recalls[:-1], label="recall")
    plt.legend()
    plt.show()

    # plot_precision_recall_curve(clf, X, y)
    # plt.show()


def find_best_threshold(clf, X, y):
    # y_scores = clf.decision_function(X)
    y_scores = cross_val_predict(clf, X, y, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls)
    best_f1_score_idx = np.argmax(f1_scores)
    print(
        f"Best threshold {thresholds[best_f1_score_idx]} - precision {precisions[best_f1_score_idx]}, recall {recalls[best_f1_score_idx]}")
    return thresholds[best_f1_score_idx]


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    tweets = pd.read_csv("data/disaster_tweets/train.csv")
    target = tweets["target"]
    tweets = tweets.drop("target", axis=1)
    for train_idx, test_idx in StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=33).split(tweets, target):
        tweets_train = tweets.iloc[train_idx]
        y_train = target.iloc[train_idx].to_numpy()
        tweets_test = tweets.iloc[test_idx]
        y_test = target.iloc[test_idx].to_numpy()

    transformer = make_column_transformer(
        (make_pipeline(TextPreprocessor(), CountVectorizer()), "text"),
        (OneHotEncoder(handle_unknown="ignore"), ["keyword"])
    )

    X_train = transformer.fit_transform(tweets_train)#.toarray()
    X_test = transformer.transform(tweets_test)

    count_vectorizer = transformer.named_transformers_.pipeline.steps[1][1]
    count_vectorizer_feature_names = count_vectorizer.get_feature_names()

    # clf = logistic_regression_grid_search(X_train, y_train, count_vectorizer_feature_names)
    # clf = knn_grid_search(X_train, y_train)
    # clf = nn_grid_search(X_train, y_train)
    # clf = svm_rbf_grid_search(X_train, y_train)
    clf = SVC(C=2.5, gamma=0.05)
    clf.fit(X_train, y_train)

    # analyse_errors_cross_val(clf, X_train, y_train)


    y_test_predicted = clf.predict(X_test)
    print(f"test accuracy: {accuracy_score(y_test, y_test_predicted)}")

    y_test_threshold_predicted = clf.decision_function(X_test) > find_best_threshold(clf, X_train, y_train)
    print(f"test accuracy with best threshold: {accuracy_score(y_test, y_test_threshold_predicted)}")

    # refit with whole training set
    X = transformer.fit_transform(tweets)
    y = target
    clf.fit(X, y)

    best_threshold = find_best_threshold(clf, X, y)
    print(f"Best threshold for whole train set: {best_threshold}")

    predict_test(clf, transformer, best_threshold)






