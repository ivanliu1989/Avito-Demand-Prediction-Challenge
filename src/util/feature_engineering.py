import pandas as pd
import numpy as np
from tqdm import tqdm
import gc


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
    dat = dat.sort_values(by=['item_id', 'date_from'])

    for c in tqdm(cols):
        print("Normal Date Transformation - {0}...".format(c))
        dat[c + '_dayofweek'] = dat[c].dt.weekday_name
        dat[c + '_dayofmonth'] = dat[c].dt.day
        dat[c + '_weekend'] = np.where(dat[c + '_dayofweek'].isin(['Saturday', 'Sunday']), 1, 0)

        print("Lagged Features - {0}...".format(c))
        dat[c + '_lag'] = dat.groupby(['item_id'])[c].shift(1)

        # print("Aggregated Features - {0}...".format(c))
        # dat[c + '_cnt'] = dat.groupby(['item_id'])[c].count()
        # dat[c + '_max'] = dat.groupby(['item_id'])[c].agg(max)
        # dat[c + '_min'] = dat.groupby(['item_id'])[c].agg(min)

    print("Activation & Promotion Cnt...")
    dat['activated'] = np.where(dat.activation_date.isna(), 0, 1)
    dat['promotion_cnt'] = dat['item_id'].groupby(dat['item_id']).transform('count')
    dat['activated_cnt'] = dat['activated'].groupby(dat['item_id']).transform('sum')

    print("Derived Features - promotion_periods...")
    dat['promotion_periods'] = dat['date_to'] - dat['date_from']
    dat.promotion_periods = dat.promotion_periods.dt.days
    dat['promotion_lifetime'] = dat['promotion_periods'].groupby(dat['item_id']).transform('sum')

    print("Derived Features - activation_gap...")
    dat['activation_gap'] = dat['date_from'] - dat['activation_date']
    dat.activation_gap = dat.activation_gap.dt.days

    print("Derived Features - days_since_last_promotion...")
    dat['days_since_last_promotion'] = dat['date_from'] - dat['date_to_lag']
    dat.days_since_last_promotion = dat.days_since_last_promotion.dt.days
    dat.days_since_last_promotion = dat.days_since_last_promotion.fillna(-1)

    # print("Derived Features - total_promotion_periods...")
    # dat['total_promotion_periods'] = dat.groupby(['item_id'])['promotion_periods'].sum()
    #
    # print("Derived Features - avg_promotion_periods...")
    # dat['avg_promotion_periods'] = dat.groupby(['item_id'])['promotion_periods'].mean()

    print("Dropping Columns not required...")
    dat = dat.drop(
        ['activation_date_lag',  # 'activation_date_max', 'activation_date_min', 'activation_date_cnt'
         'date_from_lag', 'date_to_lag'  # , 'date_to_cnt', 'date_from_cnt'
         # 'date_to_max', 'date_to_min', 'date_from_min', 'date_from_max'
         ], axis=1)

    gc.collect()

    return dat
