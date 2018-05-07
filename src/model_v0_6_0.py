import os;

os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from nltk.corpus import stopwords


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['title'] = df['title'].fillna('')
    df['param'] = df['param_1'].fillna('') + ' ' + df['param_2'].fillna('') + ' ' + df['param_3'].fillna('')
    df['text'] = (df['description'].fillna('') + ' ' + df['title'] + ' ' + df['param'].fillna(''))
    return df[['title', 'text', 'param']]


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')


def fit_predict(X_train, X_test, y_train) -> np.ndarray:
    # X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.models.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2 ** (11 + i), epochs=1, verbose=0)

        y_pred = model.predict(X_test)[:, 0]
        return y_pred


stopWords = stopwords.words('russian')
vectorizer = make_union(
    on_field('title', Tfidf(max_features=100000, stop_words=stopWords)),  # token_pattern='\w+',
    on_field('text', Tfidf(max_features=100000, ngram_range=(1, 2), stop_words=stopWords)),
    on_field('param', Tfidf(max_features=1000, stop_words=stopWords)),
    # on_field(['shipping', 'item_condition_id'],
    #          FunctionTransformer(to_records, validate=False), DictVectorizer()),
    n_jobs=4)
y_scaler = StandardScaler()
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
cv = KFold(n_splits=5, shuffle=True, random_state=42)
train_ids, valid_ids = next(cv.split(train))
train, valid = train.iloc[train_ids], train.iloc[valid_ids]
# y_train = y_scaler.fit_transform(np.log1p(train['deal_probability'].values.reshape(-1, 1)))
y_train = y_scaler.fit_transform(train['deal_probability'].values.reshape(-1, 1))
X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
print(f'X_train: {X_train.shape} of {X_train.dtype}')
del train

X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
# X_valid = vectorizer.transform(preprocess(test)).astype(np.float32)

Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]

y_pred = fit_predict(X_train, X_valid, y_train)
y_pred2 = fit_predict(Xb_train, Xb_valid, y_train)

y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
y_pred2 = np.expm1(y_scaler.inverse_transform(y_pred2.reshape(-1, 1))[:, 0])

from model.run_lightGBM import get_model_dataset, run_lightGBM, make_submission,submission_blending
print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_error(valid['deal_probability'], y_pred))))

# res = make_submission(test.item_id, y_pred2, filename='tensorflow_binary')