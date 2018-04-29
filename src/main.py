from util import target_encoding
from model.run_lightGBM import get_model_dataset, run_lightGBM, make_submission
from util.feature_engineering import load_ads_data, basic_feature_engineering
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

pd.options.display.max_columns = 999

if __name__ == '__main__':
    ### 1. Feature Engineering for Ads Periods
    # # dat.groupby('method')['year'].describe().unstack()
    # ads_periods = load_ads_data()  # (30412334, 5)
    # ads_periods = generate_ads_features(ads_periods, ["activation_date", "date_from", "date_to"])  # (30412334, 29)
    # ads_periods.head()
    # # ads_periods[np.in1d(ads_periods.item_id, ['002ec0125215', '006c6117bebc', '01f6e956b3c5'])]
    # # ads_periods[ads_periods.days_since_last_promotion < -1]
    # ads_periods.to_csv("../data/periods_all_v1.csv", index = False)

    ### 2. Active file for shopping behaviour
    # ads_periods = load_ads_data("../data/periods_all_v1.csv")
    # ads_periods.head()
    # train_active = pd.read_csv("../data/train_active.csv", parse_dates=["activation_date"])  # (14129821, 15)
    # test_active = pd.read_csv("../data/test_active.csv", parse_dates=["activation_date"])  # (12824068, 15)
    # print('train data shape: ', train_active.shape)
    # print('test data shape: ', test_active.shape)
    # # (16687412, 21)
    # train_active = pd.merge(train_active, ads_periods[ads_periods.tr_te == 1],
    #                         how="left", on=["item_id", "activation_date"])  # (1503424, 37)
    # # (13724922, 21)
    # test_active = pd.merge(test_active, ads_periods[ads_periods.tr_te == 0],
    #                        how="left", on=["item_id", "activation_date"])  # (508438, 36)

    ### 3. Merge with Cstr Txns Data
    train_dat = pd.read_csv("../data/train.csv", parse_dates=["activation_date"])  # (1503424, 18)
    test_dat = pd.read_csv("../data/test.csv", parse_dates=["activation_date"])  # (508438, 17)
    print('train data shape: ', train_dat.shape)
    print('test data shape: ', test_dat.shape)
    dat = pd.concat([train_dat, test_dat], axis=0)
    print('All data shape: ', dat.shape)  # (2011862, 37)
    train_dat = basic_feature_engineering(train_dat)
    # dat = basic_feature_engineering(dat)
    dat.head()

    # feature engineering
    features = ['item_seq_number', 'price']

    # get model datasets
    train_X, train_y, val_X, val_y, test_X, test_id = get_model_dataset(train_dat, test_dat, features)

    # run model
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 32,
        "learning_rate": 0.05,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }
    pred_test_y, model, evals_result = run_lightGBM(train_X, train_y, val_X, val_y, test_X,
                                                    params=params, early_stop=180, rounds=2000)

    # make submission
    res = make_submission(test_id, pred_test_y, filename='benchmark_end_2_end')

    # agg_cols = ['region', 'city', 'parent_category_name', 'category_name',
    #             'image_top_1', 'user_type', 'item_seq_number', 'day_of_month', 'day_of_week'];
    # for c in tqdm(agg_cols):
    #     gp = train_dat.groupby(c)['deal_probability']
    #     mean = gp.mean()
    #     std = gp.std()
    #     dat[c + '_deal_probability_avg'] = dat[c].map(mean)
    #     dat[c + '_deal_probability_std'] = dat[c].map(std)
    #
    # for c in tqdm(agg_cols):
    #     gp = train_dat.groupby(c)['price']
    #     mean = gp.mean()
    #     std = gp.std()
    #     dat[c + '_price_avg'] = dat[c].map(mean)
    #     dat[c + '_price_std'] = dat[c].map(std)
    #
    # cate_cols = ['city', 'category_name', 'user_type', 'parent_category_name', 'region']
    # for c in cate_cols:
    #     dat[c] = LabelEncoder().fit_transform(dat[c].values)
    #
    # cate_cols = ['city', 'category_name', 'user_type', 'parent_category_name', 'region']
    # for c in cate_cols:
    #     trn_tf, val_tf = target_encoding.target_encode(trn_series=train_dat[c],
    #                                                    tst_series=test_dat[c],
    #                                                    target=train_dat.deal_probability,
    #                                                    min_samples_leaf=100,
    #                                                    smoothing=20,
    #                                                    noise_level=0.01)
    #     dat[c + '_tgt_encoding'] = trn_tf.append(val_tf, ignore_index=True)
    #
    # new_data = dat.drop(['user_id', 'description', 'image',  # 'parent_category_name','region',
    #                      'item_id', 'param_1', 'param_2', 'param_3', 'title', 'deal_class', 'deal_class_2',
    #                      'day_of_week_en'], axis=1)

    # Train the model
    # parameters = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': 'rmse',
    #     'num_leaves': 31,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'verbose': 50
    # }
    #
    # model = lgb.train(parameters,
    #                   tr_data,
    #                   valid_sets=va_data,
    #                   num_boost_round=2000,
    #                   early_stopping_rounds=120,
    #                   verbose_eval=50)
