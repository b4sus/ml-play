import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                   {"max_leaf_nodes": range(5, 30, 2), "max_depth": range(3, 30, 2)},
                   # {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]},
                   cv=5)
gs.fit(X_train, y_train)
print(gs.best_estimator_)

y_test_predict = gs.predict(X_test)

print(np.mean(y_test == y_test_predict))
print(f"Accuracy with single tree is {accuracy_score(y_test, y_test_predict)}")

scores = []
trees = []
for train_index, test_index in ShuffleSplit(1000, train_size=100, random_state=42).split(X_train):
    X_train_split = X_train[train_index, :]
    y_train_split = y_train[train_index]
    cls = clone(gs.best_estimator_)
    cls.fit(X_train_split, y_train_split)
    trees.append(cls)
    scores.append(accuracy_score(y_test, cls.predict(X_test)))

print(f"mean score from 1000 datasets of size 100 is {np.mean(scores)}")

predictions = np.array([tree.predict(X_test) for tree in trees])
most_common_predictions = stats.mode(predictions)
y_test_predict_trees = most_common_predictions.mode[0]

print(f"Accuracy with many trees voting is {accuracy_score(y_test, y_test_predict_trees)}")