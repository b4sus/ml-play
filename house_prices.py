from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as op
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR

import ml.learning_curves as lc
import ml.linear_regression as lire

drop_columns = ["Id", "SalePrice", "3SsnPorch", "BsmtFinSF2", "BsmtHalfBath", "MiscVal", "YrSold", "MoSold",
                "YearRemodAdd"]


def learn_manually_with_scipy(X, y, regularization_lambda):
    op_res = op.minimize(fun=lire.linear_regression_cost_gradient,
                         x0=np.zeros((X.shape[1])),
                         args=(X, y, regularization_lambda),
                         method="CG",
                         jac=True)

    learned_theta = op_res.x

    return learned_theta.reshape((-1, 1))


def learn_with_sklearn(X, y):
    # return LinearRegression().fit(X, y)
    return Ridge().fit(X, y)
    # return RandomForestClassifier(max_features=10).fit(X, y.reshape(-1))
    # return svm.SVR(kernel="linear").fit(X, y.reshape(-1))


def predict_test_houses(pipeline, estimator):
    test_houses = pd.read_csv("house_prices/test.csv")

    test_ids = pd.DataFrame(test_houses["Id"]).set_index("Id")

    preprocess_houses(test_houses)

    X_real_test = pipeline.transform(test_houses.drop(drop_columns, axis=1, errors="ignore"))
    # X_real_test = np.hstack((np.ones((X_real_test.shape[0], 1)), X_real_test))

    y_real_pred = estimator.predict(X_real_test)

    test_ids["SalePrice"] = y_real_pred

    test_ids.to_csv("house_prices/predictions.csv")


def prepare_pipeline():
    # num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
    #                              PolynomialFeatures(include_bias=False), StandardScaler())
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    quality_categories = ["NA", "Po", "Fa", "TA", "Gd", "Ex"]

    full_pipeline = make_column_transformer(
        (num_pipeline, list(train_houses_numeric_only)),
        (OneHotEncoder(
            categories=[[str(n) for n in [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]]]),
         ["MSSubClass"]),
        (OneHotEncoder(),
         ["LandContour", "Neighborhood", "Condition1", "HouseStyle", "HouseStyle"]),
        (make_pipeline(SimpleImputer(strategy="constant", fill_value="None"), OneHotEncoder()), ["MasVnrType"]),
        (make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                       OrdinalEncoder(
                           categories=[quality_categories,
                                       quality_categories,
                                       quality_categories,
                                       quality_categories,
                                       ["NA", "Grvl", "Pave"],
                                       ["IR3", "IR2", "IR1", "Reg"],
                                       # ["Low", "HLS", "Bnk", "Lvl"],
                                       ["Inside", "Corner", "CulDSac", "FR2", "FR3"],
                                       # ["Gtl", "Mod", "Sev"],
                                       ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"],
                                       ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"],
                                       ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"],
                                       # quality_categories,
                                       # quality_categories,
                                       ]), StandardScaler()),
         [
             "FireplaceQu",
             "GarageQual",
             "GarageCond",
             "PoolQC",
             "Alley",
             "LotShape",
             # "LandContour",
             "LotConfig",
             # "LandSlope",
             "BldgType",
             "RoofStyle",
             "Foundation",
             # "BsmtQual",
             # "BsmtCond",
         ]),
        (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder()),
         ["KitchenQual", "Functional", "SaleType", "MSZoning", "BsmtExposure"]),
        (make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder()),
         ["GarageType", "GarageFinish", "Fence",
          "SaleCondition"]),
        (make_pipeline(SimpleImputer(strategy="most_frequent"),
                       OrdinalEncoder(categories=[
                           quality_categories,
                           quality_categories,
                           quality_categories,
                       ]),
                       StandardScaler()), [
             "KitchenQual",
             "BsmtQual",
             "BsmtCond"
         ]),
        (OrdinalEncoder(categories=[quality_categories, quality_categories]), ["ExterQual", "ExterCond"])
    )

    return full_pipeline


def print_cv_scores(scores):
    print(f"Scores: {scores}")
    print(f"Mean: {scores.mean()}")
    print(f"Std: {scores.std()}")


def elastic_net(X, y):
    elastic_net_cv = ElasticNetCV(cv=5, random_state=15, l1_ratio=0.9)
    elastic_net_cv.fit(X, y.ravel())
    print(f"Best alpha {elastic_net_cv.alpha_}")
    return elastic_net_cv


def grid_search_ridge(X, y):
    grid_search = GridSearchCV(Ridge(), [{"alpha": [0.1, 0.3, 0.5, 1], "solver": ["sag", "cholesky"]}], cv=5,
                               scoring="neg_root_mean_squared_error",
                               return_train_score=True)
    grid_search.fit(X, y.reshape(-1))
    print(f"Best estimator {grid_search.best_estimator_}")
    print(f"Best score {-grid_search.best_score_}")
    return grid_search.best_estimator_


def nn(X, y):
    # grid_search = GridSearchCV(MLPRegressor(max_iter=10000), [{"hidden_layer_sizes": [(120)]}], cv=3)
    # grid_search.fit(X, y.ravel())
    # print(f"Best estimator {grid_search.best_estimator_}")
    # print(f"Best score {-grid_search.best_score_}")
    # return grid_search.best_estimator_
    mlp = MLPRegressor(hidden_layer_sizes=60, max_iter=20000)
    mlp.fit(X, y.ravel())
    return mlp


def grid_search_k_neighbors(X, y):
    grid_search = GridSearchCV(KNeighborsRegressor(), [{"n_neighbors": range(1, 10)}], cv=5,
                               scoring="neg_root_mean_squared_error",
                               return_train_score=True)
    grid_search.fit(X, y.reshape(-1))
    print(f"Best estimator {grid_search.best_estimator_}")
    print(f"Best score {-grid_search.best_score_}")
    return grid_search.best_estimator_


def grid_search_random_forest(X, y):
    param_grid = [
        {"n_estimators": [20, 30, 90, 110, 120], "max_features": [1, 3, 9, 15, 27]}
    ]
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="neg_root_mean_squared_error",
                               return_train_score=True)
    grid_search.fit(X, y.reshape(-1))
    print(f"Best estimator {grid_search.best_estimator_}")
    print(f"Best score {-grid_search.best_score_}")
    return grid_search.best_estimator_


def grid_search_svm(X, y):
    param_grid = [
        {"C": [60, 70, 80, 90, 100, 110], "kernel": ["rbf", "linear", "poly"]}
    ]
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring="neg_root_mean_squared_error",
                               return_train_score=True)
    grid_search.fit(X, y.reshape(-1))
    print(f"Best estimator {grid_search.best_estimator_}")
    print(f"Best score {-grid_search.best_score_}")
    return grid_search.best_estimator_


def preprocess_houses(houses):
    def nr_months_since_2006_till_sold(house):
        sold_date = np.datetime64(f"{house['YrSold']}-{house['MoSold']:02d}")
        return (sold_date - np.datetime64("2006-01")) / np.timedelta64(1, "M")

    houses["MSSubClass"] = houses["MSSubClass"].astype(str)
    houses["SoldMonths"] = houses.apply(nr_months_since_2006_till_sold, axis=1)
    houses["RemodeledAgo"] = houses["YearRemodAdd"].max() - houses["YearRemodAdd"]


def plot_learning_curves(X_train, y_train, X_test, y_test):
    rmse = partial(mean_squared_error, squared=False)

    # plt.figure(0)
    # lc.learning_curves_of_different_training_set_size(X_train, y_train, X_test, y_test, Ridge(fit_intercept=False),
    #                                                   rmse)
    plt.figure(1)
    lc.learning_curves_of_different_lambda(X_train, y_train, X_test, y_test, lambda alpha: Ridge(alpha=alpha), rmse,
                                           [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                            20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 1000, 2000, 3000])


if __name__ == "__main__":
    houses = pd.read_csv("house_prices/train.csv")

    prices = houses["SalePrice"]

    preprocess_houses(houses)

    houses = houses.drop(drop_columns, axis=1)

    # trainset.hist()
    # plt.show()

    split = StratifiedShuffleSplit(1, test_size=0.2, random_state=53)
    for train_index, test_index in split.split(houses, houses["OverallQual"]):
        train_houses = houses.iloc[train_index]
        train_prices = prices.iloc[train_index]
        test_houses = houses.iloc[test_index]
        test_prices = prices.iloc[test_index]
    # (train_houses, test_houses, train_prices, test_prices) = train_test_split(houses, prices, test_size=0.2,
    #                                                                           random_state=53)

    train_houses_numeric_only = train_houses.select_dtypes(include=np.number)

    print(train_houses_numeric_only.head())

    full_pipeline = prepare_pipeline()

    train_transformed = full_pipeline.fit_transform(train_houses)

    y_train = train_prices.to_numpy().reshape((-1, 1))
    X_train = train_transformed
    poly_f = PolynomialFeatures(degree=2, include_bias=False)
    poly_f.fit()
    # X_train = np.hstack((np.ones((X.shape[0], 1)), X_train))

    # theta_scipy = learn_manually_with_scipy(X_train, y_train, 1)

    # print(f"training set RMSE from learning manually is {lire.rmse(theta_scipy, X_train, y_train)}")

    # scores = cross_val_score(RandomForestClassifier(max_features=10), X_train, y_train.reshape(-1),
    #                          scoring="neg_mean_squared_error", cv=10)
    # print_cv_scores(np.sqrt(-scores))

    # best_estimator = elastic_net(X_train, y_train)
    best_estimator = grid_search_ridge(X_train, y_train)
    # best_estimator = nn(X_train, y_train)

    y_test = test_prices.to_numpy().reshape((-1, 1))
    X_test = full_pipeline.transform(test_houses)
    # X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    # plot_learning_curves(X_train, y_train, X_test, y_test)

    # print(f"test set RMSE from learning manually is {lire.rmse(theta_scipy, X_test, y_test)}")

    # regr = learn_with_sklearn(X, y)
    print(
        f"training set RMSE from learning with sklearn is {mean_squared_error(y_train, best_estimator.predict(X_train), squared=False)}")
    print(
        f"test set RMSE from learning with sklearn is {mean_squared_error(y_test, best_estimator.predict(X_test), squared=False)}")

    predict_test_houses(full_pipeline, best_estimator)
