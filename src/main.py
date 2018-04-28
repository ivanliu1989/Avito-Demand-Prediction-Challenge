from util import feature_engineering
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from collections import Counter

pd.options.display.max_columns = 999


def load_ads_data():
    """
    Loading and cleaning Ads Periods data
    :return:
    """
    # Load Data
    print("Reading periods_train...")
    ads_periods_train = pd.read_csv("../data/periods_train.csv",
                                    parse_dates=["activation_date", "date_from", "date_to"])

    print("Reading periods_test...")
    ads_periods_test = pd.read_csv("../data/periods_test.csv", parse_dates=["activation_date", "date_from", "date_to"])

    print("Assigning train/test flag...")
    ads_periods_train['tr_te'] = 1
    ads_periods_test['tr_te'] = 0

    print("Concatenating...")
    ads_periods = pd.concat([ads_periods_train, ads_periods_test], axis=0)
    print('Ads Periods data shape: ', ads_periods.shape)
    gc.collect()

    return ads_periods


# Data Cleaning for Ads Periods
def generate_ads_features(dat, cols):
    """
    Generating Ads Periods Features
    :param dat:
    :param cols:
    :return:
    """
    print("Sorting by dates...")
    dat = dat.sort_values(by=['item_id', 'activation_date'])

    for c in tqdm(cols):
        print("Normal Date Transformation - {0}...".format(c))
        dat[c + '_dayofweek'] = dat[c].dt.weekday_name
        dat[c + '_dayofmonth'] = dat[c].dt.day
        dat[c + '_weekend'] = np.where(dat[c + '_dayofweek'].isin(['Saturday', 'Sunday']), 1, 0)

        print("Lagged Features - {0}...".format(c))
        dat[c + '_lag'] = dat.groupby(['item_id'])[c].shift(1)

        print("Aggregated Features - {0}...".format(c))
        dat[c + '_cnt'] = dat.groupby(['item_id'])[c].count()
        dat[c + '_max'] = dat.groupby(['item_id'])[c].max()
        dat[c + '_min'] = dat.groupby(['item_id'])[c].min()

    print("Derived Features - promotion_periods...")
    dat['promotion_periods'] = dat['date_to'] - dat['date_from']

    print("Derived Features - activation_gap...")
    dat['activation_gap'] = dat['date_from'] - dat['activation_date']

    print("Derived Features - days_since_last_promotion...")
    dat['days_since_last_promotion'] = dat['date_from'] - dat['date_to_lag']

    print("Derived Features - total_promotion_periods...")
    dat['total_promotion_periods'] = dat.groupby(['item_id'])['promotion_periods'].sum()

    print("Derived Features - avg_promotion_periods...")
    dat['avg_promotion_periods'] = dat.groupby(['item_id'])['promotion_periods'].mean()

    # Drop columns
    print("Dropping Columns not required...")

    gc.collect()

    return dat


ads_periods = load_ads_data()
ads_periods = generate_ads_features(ads_periods, ["activation_date", "date_from", "date_to"])
ads_periods.head()

# Merge with Cstr Txns Data
train_dat = pd.read_csv("../data/train.csv", parse_dates=["activation_date"])
print('train data shape: ', train_dat.shape)
dat = pd.merge(train_dat, ads_periods, how="left", on=["item_id", "activation_date"])
dat.head()

# By customer/region/category etc.
# City/Category/UseType/Activation_date Ads counts / distribution / percentage
Counter(dat.param_3)
len(set(dat.image_top_1))
cols = ['category_name', 'city', 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']
