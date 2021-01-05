import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost
import pickle

# Logging configuration
MSG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=MSG_FORMAT, datefmt=DATETIME_FORMAT)
logger = logging.getLogger("2K RATING")
logger.setLevel(logging.INFO)

# Global variables
root = os.path.abspath(os.path.join(os.curdir, "../.."))

exclude = [
    "Rk", "Player", "Pos", "G", "GS",
    "FG", "3P", "2P", "FT", "ORB",
    "DRB", "AST", "STL", "BLK", "TOV", "PTS",
    "FGA", "3PA", "2PA", "FTA", "TRB", "PF"
]

target = "Rating"


def training(X_train, y_train, X_test, y_test, model_name):
    """
    """
    dt = pd.DataFrame(X_train.dtypes, columns=["type"])
    features_cat = list(dt[(dt["type"] == np.object)].index)
    features_numeric = list(dt[(dt["type"] == np.number) | (dt["type"] == np.integer)].index)

    for feat in exclude + [target]:
        if feat in features_cat:
            features_cat.remove(feat)
        if feat in features_numeric:
            features_numeric.remove(feat)

    pipe_numeric = Pipeline(steps=[
    ("Imputation", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    pipe_cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="inconnu")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        [
            ("numeric", pipe_numeric, features_numeric),
            ("categorical", pipe_cat, features_cat),
        ],
        remainder="drop"
    )

    xgb_params = {
        "objective": "reg:squarederror",
        "random_state": 23,
        "early_stopping_rounds": 50,
        'eval_metric': ['rmse', 'mae'],
        'eval_set': [[X_test, y_test]]
    }

    grid_params = {
        "regressor__n_estimators": [10, 20, 30, 50, 200, 500],
        "regressor__max_depth": [1, 2, 3, 5],  # the maximum depth of each tree
        "regressor__learning_rate": [0.01, 0.1, 0.2],  # the training step for each iteration
        "regressor__min_child_weight": [4,5],
        "regressor__gamma": [i/10.0 for i in range(3,6)]
    }

    scoring = {
        'Neg RMSE': 'neg_root_mean_squared_error',
        'Neg MAE': 'neg_mean_absolute_error'
    }

    regressor = xgboost.XGBRegressor(**xgb_params)

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    gs = GridSearchCV(
        estimator=clf,
        param_grid=grid_params,
        verbose=1,
        cv=5,
        return_train_score=True,
        n_jobs=-1,
        scoring=scoring,
        refit="Neg RMSE"
    )

    gs.fit(X_train, y_train)

    print(gs.best_score_)
    print(gs.best_params_)

    pickle.dump(gs.best_estimator_, open(model_name, 'wb'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    args = parser.parse_known_args()[0]

    logger.info(">>> Read raw data")
    df_train = pd.read_csv("{}/data/input/df_train.csv".format(root))
    df_test = pd.read_csv("{}/data/input/df_test.csv".format(root))
    df_val = pd.read_csv("{}/data/input/df_val.csv".format(root))

    X_train, X_test, X_val = df_train.drop([target], axis=1), df_test.drop([target], axis=1), df_val.drop([target], axis=1)
    y_train, y_test, y_val = df_train[target], df_test[target], df_val[target]

    logger.info(">>> Train and save the model")
    model_path = "{}/artefacts/model/model.pkl".format(root)
    training(X_train, y_train, X_test, y_test, model_path)

    logger.info(">>> Assess performance on validation set")
    reg = pickle.load(open(model_path, 'rb'))
    y_pred = reg.predict(X_val)
    print("MSE: {}".format(mean_squared_error(y_val, y_pred, squared=False)))
    print("MAE: {}".format(mean_absolute_error(y_val, y_pred)))
