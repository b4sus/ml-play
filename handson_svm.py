import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def ex_8():
    iris_ds = sklearn.datasets.load_iris()
    X = iris_ds["data"][:, (2, 3)]
    y = (iris_ds["target"] == 2).astype(np.float64)

    plt.figure(0)
    plt.plot(X[y == 1, 0], X[y == 1, 1], "xb")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "xy")

    linear_svc = LinearSVC(C=1)
    scaler = StandardScaler()
    pipeline = make_pipeline(scaler, linear_svc)
    pipeline.fit(X, y)

    x0_linspace = np.linspace(0.8, 7, 100)
    # x1_linspace = np.linspace(0, 2.6, 100)
    # Z = np.empty((100, 100))
    # for i in range(len(x0_linspace)):
    #     for j in range(len(x1_linspace)):
    #         Z[i, j] = pipeline.predict([[x0_linspace[i], x1_linspace[j]]])
    #
    # plt.contour(x0_linspace, x1_linspace, Z)

    w = linear_svc.coef_[0]
    b = linear_svc.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0_linspace = scaler.transform(np.column_stack((x0_linspace.T, np.ones((100, 1)))))[:, 0]
    decision_boundary = -w[0]/w[1] * x0_linspace - b/w[1]

    rescaled = scaler.inverse_transform(np.column_stack((x0_linspace, decision_boundary.T)))
    plt.plot(rescaled[:, 0], rescaled[:, 1], "k-", linewidth=1)
    plt.show()


if __name__ == "__main__":
    ex_8()
