import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from util.target_encoding import target_encode

pd.options.display.max_columns = 999


def load_ads_data(path=None):
    """
    Loading and cleaning Ads Periods data
    :return:
    """
    if path is None:
        # Load Data
        print("Reading periods_train...")
        # (16687412, 5)
        ads_periods_train = pd.read_csv("../data/periods_train.csv",
                                        parse_dates=["activation_date", "date_from", "date_to"])
        # (13724922, 5)
        print("Reading periods_test...")
        ads_periods_test = pd.read_csv("../data/periods_test.csv",
                                       parse_dates=["activation_date", "date_from", "date_to"])

        print("Assigning train/test flag...")
        ads_periods_train['tr_te'] = 1
        ads_periods_test['tr_te'] = 0

        print("Concatenating...")
        ads_periods = pd.concat([ads_periods_train, ads_periods_test], axis=0)
        print('Ads Periods data shape: ', ads_periods.shape)
        gc.collect()

    else:
        ads_periods = pd.read_csv(path, parse_dates=["activation_date", "date_from", "date_to"])

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


def transform_date(dat, cols=None):
    """

    :param dat:
    :param cols:
    :return:
    """
    for c in tqdm(cols):
        # print("Sorting by dates...")
        # dat = dat.sort_values(by=[key, c])
        print("Normal Date Transformation - {0}...".format(c))
        dat[c + '_dayofweek'] = dat[c].dt.weekday_name
        dat[c + '_dayofmonth'] = dat[c].dt.day
        dat[c + '_weekend'] = np.where(dat[c + '_dayofweek'].isin(['Saturday', 'Sunday']), 1, 0)

        # print("Lagged Features - {0}...".format(c))
        # dat[c + '_lag'] = dat.groupby([key])[c].shift(1)

    return dat


def target_encoding(dat, tgt_cols, cate_cols, measure=['mean'], noise=True):
    if noise:
        dat
        # target_encode
    else:
        for t in tqdm(tgt_cols):
            for c in tqdm(cate_cols):
                for m in measure:
                    # TODO(Ivan): Add multi categories
                    f = '{0}_{1}_{2}'.format(('').join(c), t, m)
                    print(f)
                    # dat[f] = dat[t].groupby(dat[c]).transform(m)
                    if m == 'kurt':
                        dat[f] = dat.groupby(c)[t].apply(pd.DataFrame.kurt)
                    else:
                        dat[f] = dat.groupby(c)[t].transform(m)

    return dat


def feature_engineering_v1(train_dat, test_dat):
    """

    :param train_dat:
    :param test_dat:
    :return:
    """
    print('train data shape: ', train_dat.shape)
    print('test data shape: ', test_dat.shape)
    train_dat['tr_te'] = 1
    test_dat['tr_te'] = 0
    dat = pd.concat([train_dat, test_dat], axis=0)
    print('All data shape: ', dat.shape)  # (2011862, 37)

    # Activation Date
    dat = transform_date(dat, ['activation_date'])

    # Deal Probability
    dat['deal_class'] = dat['deal_probability'].apply(lambda x: ">=0.5" if x >= 0.5 else "<0.5")
    interval = (-0.99, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.1)
    cats = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    dat["deal_class_2"] = pd.cut(dat.deal_probability, interval, labels=cats)

    # Title
    dat['title'] = dat['title'].fillna(" ")
    dat['title_len'] = dat['title'].apply(lambda x: len(x.split()))

    # Description
    dat['description'] = dat['description'].fillna(" ")
    dat['description_len'] = dat['description'].apply(lambda x: len(x.split()))

    # Target Mean
    tgt_cols = ['deal_probability', 'price', 'image_top_1', 'activation_date_weekend']
    cate_cols = ['category_name', 'region', 'city', 'param_1', 'param_2',
                 'parent_category_name', 'user_type', 'activation_date_dayofweek',
                 'deal_class', 'deal_class_2',
                 ['activation_date_dayofweek', 'region']]
    measures = ['mean', 'std', 'quantile', 'skew', 'count'] # , 'kurt'
    dat = dat.sort_values(by=['activation_date'])
    dat = target_encoding(dat, tgt_cols, cate_cols, measures, False)

    # Fill NA
    # dat.fillna(-1)

    # Split train & test
    train_dat = dat[dat.tr_te == 1]
    test_dat = dat[dat.tr_te == 0]

    dat.head()
    return train_dat, test_dat
