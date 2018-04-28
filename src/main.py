from util import feature_engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
from collections import Counter

train_dat = pd.read_csv("../data/train.csv", parse_dates=["activation_date"])
ads_periods = pd.read_csv("../data/periods_train.csv", parse_dates=["activation_date", "date_from", "date_to"])
print('train data shape: ', train_dat.shape)
print('ads periods data shape: ', ads_periods.shape)
gc.collect()


# Data Cleaning for Ads Periods
def get_day_features(dat, cols):
    for c in tqdm(cols):
        dat[c + '_dayofweek'] = dat[c].dt.weekday_name
        dat[c + '_dayofmonth'] = dat[c].dt.day
        dat[c + '_weekend'] = np.where(dat[c + '_dayofweek'].isin(['Saturday', 'Sunday']), 1, 0)
        dat[c + '_lag'] = dat.groupby(['item_id'])[c].shift(1)

    return dat


ads_periods.head()
ads_periods = get_day_features(ads_periods, ["activation_date", "date_from", "date_to"])

















# Feature engineering for Ads Periods
def ads_periods_feature_engineering(dat):
    return dat


# Basic date gap & lagged features
ads_periods['promotion_periods'] = ads_periods['date_to'] - ads_periods['date_from']
ads_periods['activation_gap'] = ads_periods['date_from'] - ads_periods['activation_date']



# Aggregated features
ads_periods_grp = ads_periods.groupby('item_id')
ads_periods_grouped = ads_periods_grp.agg({"activation_date": ["nunique", "min", "max"],
                                           "date_from": ["nunique", "min", "max"],
                                           "date_to": ["nunique", "min", "max"]
                                           })
ads_periods_grouped.columns = ["_".join(x) for x in ads_periods_grouped.columns.ravel()]
ads_periods_grouped.head()
ads_periods_grouped['activation_date_gap'] = ads_periods_grouped['activation_date_max'] - ads_periods_grouped['activation_date_min']
ads_periods_grouped['promotion_gap'] = ads_periods_grouped['date_from_max'] - ads_periods_grouped['date_to_min']


Counter(ads_periods_grouped.activation_date_nunique)












dat = pd.merge(train_dat, ads_periods, how="left", on=["item_id", "activation_date"])
dat.head()

# City/Category/UseType/Activation_date Ads counts / distribution / percentage
Counter(dat.param_3)
len(set(dat.image_top_1))
cols = ['category_name', 'city', 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']
