import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb


def get_model_dataset(train_df, test_df, features, val_date='2017-03-23'):
    """

    :param train_df:
    :param test_df:
    :param features:
    :param val_date:
    :return:
    """
    # Target and ID variables #
    train_y = train_df["deal_probability"].values
    test_id = test_df["item_id"].values

    dev_X = train_df[train_df.activation_date < val_date]
    dev_y = train_y[train_df.activation_date < val_date]
    val_X = train_df[train_df.activation_date >= val_date]
    val_y = train_y[train_df.activation_date >= val_date]

    dev_X = dev_X[features]
    val_X = val_X[features]
    test_X = test_df[features]

    return dev_X, dev_y, val_X, val_y, test_X, test_id


def run_lightGBM(train_X, train_y, val_X, val_y, test_X, params=None, early_stop=100, rounds=1000):
    """

    :param train_X:
    :param train_y:
    :param val_X:
    :param val_y:
    :param test_X:
    :param params:
    :param early_stop:
    :param rounds:
    :return:
    """
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 30,
            "learning_rate": 0.1,
            "bagging_fraction": 0.7,
            "feature_fraction": 0.7,
            "bagging_frequency": 5,
            "bagging_seed": 2018,
            "verbosity": -1
        }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, rounds, valid_sets=[lgval], early_stopping_rounds=early_stop, verbose_eval=20,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result


def make_submission(test_id, pred_test, filename='benchmark_1'):
    """

    :param test_id:
    :param pred_test:
    :param filename:
    :return:
    """
    sub_df = pd.DataFrame({"item_id": test_id})
    sub_df["deal_probability"] = pred_test
    filename = "../submissions/{}.csv".format(filename)
    sub_df.to_csv(filename, index=False)
    print('Predictions saved into: ' + filename)

    return filename
