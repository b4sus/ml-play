import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def split(students, labels):
    split = StratifiedShuffleSplit(1, test_size=0.2, random_state=15)
    for students_train_idx, students_test_idx in split.split(students, students["sex"]):
        return students.iloc[students_train_idx], labels.iloc[students_train_idx], students.iloc[students_test_idx], labels.iloc[students_test_idx]


def prepare_pipeline(students):
    # numerical_pipeline = make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(degree=2), StandardScaler())
    numerical_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    bool_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder())
    return make_column_transformer(
        (numerical_pipeline, make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(categories=[["T", "A"]]), ["Pstatus"]),
        (OneHotEncoder(categories=[["U", "R"]]), ["address"]),
        (OneHotEncoder(categories=[["F", "M"]]), ["sex"]),
    )


def yes_no_to_bool(yes_no):
    if yes_no is None:
        return None
    elif yes_no == "yes":
        return True
    elif yes_no == "no":
        return False
    else:
        raise ValueError(f"'yes' or 'no' or None expected, got {yes_no}")


def preprocess(students):
    students = students.copy()
    for column in students.columns:
        if {"yes", "no"} == set(np.unique(students[column].to_numpy())):
            students[column] = students[column].apply(yes_no_to_bool)

    for bool_col in students.select_dtypes(include=bool).columns:
        students[bool_col] = students[column].apply(lambda true_false: 1 if true_false == True else 0)
    # students.drop(["Dalc", "Walc"], axis=1, inplace=True)
    students.drop(["Walc"], axis=1, inplace=True)
    return students


def evaluate_performance(clf, X_train, y_train):
    y_train_5 = y_train == 5

    print(cross_val_score(clone(clf), X_train, y_train_5, cv=3, scoring="accuracy"))

    y_train_5_predicted = cross_val_predict(clone(clf), X_train, y_train_5, cv=3)

    print(confusion_matrix(y_train_5, y_train_5_predicted))

    print(f"precision score {precision_score(y_train_5, y_train_5_predicted)}")
    print(f"recall score {recall_score(y_train_5, y_train_5_predicted)}")
    print(f"f1 score {f1_score(y_train_5, y_train_5_predicted)}")

    y_scores = cross_val_predict(clf, X_train, y_train_5, method="decision_function", cv=3)

    # fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    # plot_roc_curve(fpr, tpr)

    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    plot_precision_recall_curve(precisions, recalls)


def plot_precision_recall_curve(precisions, recalls):
    plt.plot(recalls, precisions)
    plt.show()


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()


def analyze_errors(clf, X, y):
    y_pred = cross_val_predict(clf, X, y, cv=3)
    conf_mx = confusion_matrix(y, y_pred)

    print(conf_mx)

    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    students = pd.read_csv("data/students/student-mat.csv")
    alcohol_consumptions = students[["Dalc", "Walc"]]

    students = preprocess(students)

    # print(alcohol_consumptions)

    students_train, labels_train, students_test, labels_test = split(students, alcohol_consumptions)

    print(students_train.info())

    pipeline = prepare_pipeline(students_train)

    X_train = pipeline.fit_transform(students_train)
    y_train = labels_train["Walc"].to_numpy()

    # clf = GridSearchCV(LogisticRegression(max_iter=1000), param_grid={"C": [0.01, 0.1, 0.3, 1, 3, 6, 9],
    #                    }, cv=5)
    clf = GridSearchCV(SVC(), param_grid={"C": [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 1, 3, 6, 9], "kernel": ["linear", "poly"]}, cv=3)
    # clf = GridSearchCV(RandomForestClassifier(), param_grid={"n_estimators": [10, 20, 50, 100], "max_depth": range(1, 20)}, cv=3, )

    # clf = LogisticRegression(max_iter=1000, C=0.01)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    y_train_predicted = clf.predict(X_train)
    # y_train_predicted = cross_val_predict(cls, X, labels_train["Walc"].to_numpy(), cv=5)
    print(f"train accuracy {accuracy_score(y_train, y_train_predicted)}")

    print(cross_val_score(clone(clf.best_estimator_), X_train, y_train, cv=3, scoring="accuracy"))


    # evaluate_performance(clone(clf.best_estimator_), X_train, y_train)

    analyze_errors(clf, X_train, y_train)


    y_test = labels_test["Walc"].to_numpy()
    y_test_predicted = clf.predict(pipeline.transform(students_test))
    # print(f"test accuracy {accuracy_score(y_test, y_test_predicted)}")


    # print(f"{students.describe()}")
    # students["age"].hist()
    # plt.show()
