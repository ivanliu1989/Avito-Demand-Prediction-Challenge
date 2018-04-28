from util.feature_engineering import load_ads_data, generate_ads_features
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


    ### 3. Merge with Cstr Txns Data
    ads_periods = load_ads_data("../data/periods_all_v1.csv")
    ads_periods.head()
    train_dat = pd.read_csv("../data/train.csv", parse_dates=["activation_date"]) # (1503424, 18)
    test_dat = pd.read_csv("../data/test.csv", parse_dates=["activation_date"]) # (508438, 17)
    print('train data shape: ', train_dat.shape)
    print('test data shape: ', test_dat.shape)
    # (16687412, 21)
    train_dat = pd.merge(train_dat, ads_periods[ads_periods.tr_te == 1],
                         how="left", on=["item_id", "activation_date"]) # (1503424, 37)
    # (13724922, 21)
    test_dat = pd.merge(test_dat, ads_periods[ads_periods.tr_te == 0],
                        how="left", on=["item_id", "activation_date"]) # (508438, 36)
    dat.head()

    # By customer/region/category etc.
    # City/Category/UseType/Activation_date Ads counts / distribution / percentage
    Counter(dat.param_3)
    len(set(dat.image_top_1))
    cols = ['category_name', 'city', 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']

    # Train the model
    parameters = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 50
    }

    model = lgb.train(parameters,
                      tr_data,
                      valid_sets=va_data,
                      num_boost_round=2000,
                      early_stopping_rounds=120,
                      verbose_eval=50)
