from util import target_encoding
from model.run_lightGBM import get_model_dataset, run_lightGBM, make_submission
from util.feature_engineering import load_ads_data, feature_engineering_v1
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt

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

    # feature engineering
    train_df, test_df = feature_engineering_v1(train_dat, test_dat)
    gc.collect()
    train_df.head()
    features = ['image_top_1','item_seq_number', 'price','activation_date_dayofmonth','activation_date_weekend',
                'title_len', 'description_len', 'region_deal_probability_mean','region_deal_probability_median',
                'city_deal_probability_mean', 'city_deal_probability_median', 'activation_date_dayofweekregion_deal_probability_mean',
                'activation_date_dayofweekregion_deal_probability_median','region_price_mean','region_price_median',
                'city_price_mean','city_price_median','activation_date_dayofweekregion_price_mean','activation_date_dayofweekregion_price_median']

    # get model datasets
    train_X, train_y, val_X, val_y, test_X, test_id = get_model_dataset(train_df, test_df, features)

    ### 4. run model
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }
    pred_test_y, model, evals_result = run_lightGBM(train_X, train_y, val_X, val_y, test_X,
                                                    params=params, early_stop=180, rounds=2000)
    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=15)
    plt.show()

    ### 5. make submission
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
