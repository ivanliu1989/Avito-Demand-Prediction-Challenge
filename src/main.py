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
    features_to_drop = ['activation_date', 'category_name', 'city', 'deal_probability', 'description',
                        'image', 'item_id', 'param_1', 'param_2', 'param_3', 'parent_category_name',
                        'region', 'title', 'tr_te', 'user_id', 'user_type', 'activation_date_dayofweek',
                        'deal_class', 'deal_class_2']
    # features = ['image_top_1','item_seq_number', 'price','activation_date_dayofmonth','activation_date_weekend',
    #             'title_len', 'description_len', 'region_deal_probability_mean','region_deal_probability_median',
    #             'city_deal_probability_mean', 'city_deal_probability_median', 'activation_date_dayofweekregion_deal_probability_mean',
    #             'activation_date_dayofweekregion_deal_probability_median','region_price_mean','region_price_median',
    #             'city_price_mean','city_price_median','activation_date_dayofweekregion_price_mean','activation_date_dayofweekregion_price_median']

    # get model datasets
    train_X, train_y, val_X, val_y, test_X, test_id = get_model_dataset(train_df, test_df, features_to_drop,
                                                                        val_date='2017-03-27')
    train_X.head()

    ### 4. run model
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.5,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    pred_test_y, model, evals_result = run_lightGBM(train_X, train_y, val_X, val_y, test_X,
                                                    params=params, early_stop=100, rounds=5000)

    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=15)
    plt.show()

    ### 5. make submission
    res = make_submission(test_id, pred_test_y, filename='v0_0_0_1_val_0_225643_3')


    ### 6. blending
    paths = ['../submissions/v0_0_0_1_val_0_225643_2.csv',
             '../submissions/baseline_lgb_0_229.csv',
             '../submissions/v0_0_0_1_val_0_225768.csv']
    wts = [0.5, 0.3, 0.2]