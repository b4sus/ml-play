import csv
import random

import numpy as np
import ml.logistic_regression as lori
import ml.predict as predict
import ml.feature as feature
import ml.pipeline as pline
import scipy.optimize as op
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

np.seterr(all="raise")

embarked_mapping = {"C": 1, "Q": 2, "S": 3, }

class Passenger:
    def __init__(self, param_dict):
        self.id = param_dict['PassengerId']
        self.name = param_dict['Name']
        if param_dict['Age']:
            self.age = float(param_dict['Age'])
        else:
            self.age = float(40)

        self.ticket_class = int(param_dict['Pclass'])
        self.sex = float(-1) if param_dict['Sex'] == 'male' else float(1)
        self.num_siblings_spouses = int(param_dict['SibSp'])
        self.num_parents_children = int(param_dict['Parch'])
        self.ticket_nr = param_dict['Ticket']
        self.fare = float(param_dict['Fare']) if param_dict['Fare'] else 15.
        self.cabin = param_dict['Cabin']
        embarked = param_dict['Embarked']
        if embarked:
            self.embarked = embarked_mapping[embarked]
        else:
            self.embarked = 2

    def __str__(self):
        return f'{self.id} - {self.name} - {self.survived}'


class TrainPassenger(Passenger):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.survived = bool(int(param_dict['Survived']))

    def __str__(self):
        return f"{self.id} - {self.name}"


passengers = []

with open('data/titanic-train.csv', 'r') as train_csv:
    dict_reader = csv.DictReader(train_csv)
    for row in dict_reader:
        passengers.append(TrainPassenger(row))

random.seed(3)
random.shuffle(passengers)

# passengers = [p for p in passengers if p.fare < 500]  # really huge (well above average)
# entries can cause division by 0 when they are multiplied by polynomial features

train_passengers = passengers[:int(len(passengers) * 0.6)]
cv_passengers = passengers[int(len(passengers) * 0.6):int(len(passengers) * 0.8)]
test_passengers = passengers[int(len(passengers) * 0.8):]


def create_feature_matrix(passengers):
    X = np.empty((0, 6))
    for (idx, passenger) in enumerate(passengers):
        X = np.vstack((X, [passenger.sex, passenger.age, passenger.ticket_class, passenger.fare,
                           passenger.num_siblings_spouses, passenger.num_parents_children]))
    return X


def create_result_vector(passengers):
    y = np.empty((len(passengers), 1))
    for (idx, passenger) in enumerate(passengers):
        y[idx, 0] = 1.0 if passenger.survived else 0.0
    return y


X_train = create_feature_matrix(train_passengers)
y_train = create_result_vector(train_passengers)

X_cv = create_feature_matrix(cv_passengers)
y_cv = create_result_vector(cv_passengers)

j_train = []
j_cv = []
polynomial_degrees = list(range(1, 10))
for polynomial_degree in polynomial_degrees:
    pipeline_pd = pline.Pipeline()
    pipeline_pd.one_hot_encode([2])
    pipeline_pd.polynomial(polynomial_degree, include_bias=False, interaction_only=True)
    pipeline_pd.reduce_features_without_std()
    pipeline_pd.normalize()
    pipeline_pd.bias()
    (theta, X_processed) = pipeline_pd.execute_train(X_train, y_train, regularization_lambda=3)

    j_train.append(lori.logistic_regression_cost(X_processed, y_train, theta))
    j_cv.append(lori.logistic_regression_cost(pipeline_pd.process_test(X_cv), y_cv, theta))

plt.figure(0)
plt.plot(polynomial_degrees, j_train, label="j_train")
plt.plot(polynomial_degrees, j_cv, label="j_cv")
plt.xlabel("polynomial degree")
plt.legend()
plt.show(block=False)

pipeline = pline.Pipeline()
pipeline.one_hot_encode([2])
pipeline.polynomial(3, include_bias=False, interaction_only=True)
pipeline.reduce_features_without_std()
pipeline.normalize()
pipeline.bias()


lambdas = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 8, 10.24, 13, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# lambdas = np.arange(0, 5, 0.1)

j_train = []
j_cv = []

for regularization_lambda in lambdas:
    (theta, X_processed) = pipeline.execute_train(X_train, y_train, regularization_lambda=regularization_lambda)

    j_train.append(lori.logistic_regression_cost(X_processed, y_train, theta))
    j_cv.append(lori.logistic_regression_cost(pipeline.process_test(X_cv), y_cv, theta))

plt.figure(1)
plt.plot(lambdas, j_train, label="j_train")
plt.plot(lambdas, j_cv, label="j_cv")
plt.xlabel("lambda")
plt.legend()
plt.show()

(theta, X_processed) = pipeline.execute_train(X_train, y_train, regularization_lambda=1)

j_train = lori.logistic_regression_cost(X_processed, y_train, theta)

print(f"j_train: {j_train}")

predictions = predict.predict(X_processed[:, 1:], theta, lori.logistic_regression_hypothesis)

print(f"train accuracy: {np.mean(predictions == y_train)}")

predictions_validation = predict.predict(pipeline.process_test(X_cv)[:, 1:], theta, lori.logistic_regression_hypothesis)

print(f"cv accuracy: {np.mean(predictions_validation == y_cv)}")

X_test = create_feature_matrix(test_passengers)
y_test = create_result_vector(test_passengers)

predictions_test = predict.predict(pipeline.process_test(X_test)[:, 1:], theta, lori.logistic_regression_hypothesis)

print(f"test accuracy: {np.mean(predictions_test == y_test)}")

with open('data/titanic-test.csv', 'r') as test_csv:
    dict_reader = csv.DictReader(test_csv)
    for row in dict_reader:
        test_passengers.append(Passenger(row))

X_test = create_feature_matrix(test_passengers)

X_test = pipeline.process_test(X_test)

predictions_test = predict.predict(X_test[:, 1:], theta, lori.logistic_regression_hypothesis)

# print(predictions_test)

results = zip([p.id for p in test_passengers], predictions_test.flatten())

# for zipped in results:
#     print(zipped)

with open("titanic-result.csv", "w", newline='') as result_csv:
    writer = csv.writer(result_csv, delimiter=",")
    writer.writerow(("PassengerId", "Survived"))
    for result in results:
        writer.writerow(result)
