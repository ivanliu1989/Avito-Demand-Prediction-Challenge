import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
from nltk.corpus import stopwords
from tqdm import tqdm
from model.run_lightGBM import get_model_dataset, run_lightGBM, make_submission, submission_blending

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
# Any results you write to the current directory are saved as output.

stopWords = stopwords.words('russian')

train_df = pd.read_csv("../data/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../data/test.csv", parse_dates=["activation_date"])

# Target and ID variables #
train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values

### R
train_df.price = np.log(train_df.price)
test_df.price = np.log(test_df.price)

train_df.city = train_df.city.fillna(' ')
train_df.param_1 = train_df.param_1.fillna(' ')
train_df.param_2 = train_df.param_2.fillna(' ')
train_df.param_3 = train_df.param_3.fillna(' ')
train_df.title = train_df.title.fillna(' ')
train_df.description = train_df.description.fillna(' ')
test_df.city = test_df.city.fillna(' ')
test_df.param_1 = test_df.param_1.fillna(' ')
test_df.param_2 = test_df.param_2.fillna(' ')
test_df.param_3 = test_df.param_3.fillna(' ')
test_df.title = test_df.title.fillna(' ')
test_df.description = test_df.description.fillna(' ')

train_df.txt = train_df.city + ' ' + train_df.param_1 + ' ' + train_df.param_2 + ' ' + train_df.param_3 + ' ' + train_df.title + ' ' + train_df.description
test_df.txt = test_df.city + ' ' + test_df.param_1 + ' ' + test_df.param_2 + ' ' + test_df.param_3 + ' ' + test_df.title + ' ' + test_df.description

train_df.mday = train_df.activation_date.dt.day
test_df.mday = test_df.activation_date.dt.day
train_df.wday = train_df.activation_date.dt.weekday
test_df.wday = test_df.activation_date.dt.weekday

train_df.image_top_1 = train_df.image_top_1.fillna(-1)
test_df.image_top_1 = test_df.image_top_1.fillna(-1)
train_df.price = train_df.price.fillna(-1)
test_df.price = test_df.price.fillna(-1)

train_df.txt = train_df.txt.str.lower()
test_df.txt = test_df.txt.str.lower()

train_df.txt = train_df.txt.str.replace("[^[:alpha:]]", " ")
test_df.txt = test_df.txt.str.replace("[^[:alpha:]]", " ")
train_df.txt = train_df.txt.str.replace("\\s+", " ")
test_df.txt = test_df.txt.str.replace("\\s+", " ")

topWords = stopwords.words('russian')

# Create tfidf matrix for title and description
tfidf = TfidfVectorizer(max_features=5000, stop_words=stopWords, ngram_range=(1, 3),
                        min_df=3, max_df=0.3, norm='l2', sublinear_tf=True)
tfidf.fit(pd.concat([train_df.txt, test_df.txt]))
train_df_tfidf = tfidf.transform(train_df.txt)
test_df_tfidf = tfidf.transform(test_df.txt)
train_df_tfidf = pd.SparseDataFrame(train_df_tfidf, columns=tfidf.get_feature_names(), default_fill_value=0)
test_df_tfidf = pd.SparseDataFrame(test_df_tfidf, columns=tfidf.get_feature_names(), default_fill_value=0)

train_df_tfidf.columns = ['tfidf_' + str(i) for i in range(len(train_df_tfidf.columns))]
test_df_tfidf.columns = ['tfidf_' + str(i) for i in range(len(test_df_tfidf.columns))]

cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image",
                'region','city', 'parent_category_name','category_name','param_1','param_2','param_3','user_type']

train_dt = pd.concat([train_df.drop(cols_to_drop, axis=1).to_sparse(fill_value=0), train_df_tfidf], axis=1)
test_dt = pd.concat([test_df.drop(cols_to_drop, axis=1).to_sparse(fill_value=0), test_df_tfidf], axis=1)

# train_X, train_y, val_X, val_y, test_X, test_id = get_model_dataset(train_dt, test_dt, [],
#                                                                     val_date='2017-03-27')
# train_X, val_X, train_y, val_y = model_selection.train_test_split(train_dt,
#                                                                   train_dt.deal_probability,
#                                                                   test_size=0.1, random_state=19)

train_X = train_dt.iloc[:-200000, :]
val_X = train_dt.iloc[-200000:, :]
train_y = train_dt.deal_probability[:-200000]
val_y = train_dt.deal_probability[-200000:]

train_X = train_X.drop(["deal_probability"], axis=1)
val_X = val_X.drop(["deal_probability"], axis=1)
train_X.head()

### 4. run model
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 30,  # 40
    "learning_rate": 0.1,  # 0.09
    "bagging_fraction": 0.7,
    "feature_fraction": 0.7,
    "bagging_frequency": 5,
    "bagging_seed": 2018,
    "verbosity": -1
    # ,"boosting":"dart" # gbdt, rf, dart
    # ,"device":"gpu"
}

pred_test_y, model, evals_result, cv_results = run_lightGBM(train_X, train_y, val_X, val_y, test_X,
                                                            params=params, early_stop=100, rounds=5000)





# fit model no training data
import xgboost as xgb

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['eval_metric'] = 'auc'
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 8
num_rounds = 5000

plst = list(param.items())
xgtrain = xgb.DMatrix(train_X, label=train_y)
xgval = xgb.DMatrix(val_X, label=val_y)
xgtest = xgb.DMatrix(test_X)
watchlist = [(xgtrain, 'train'), (xgval, 'validation')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
