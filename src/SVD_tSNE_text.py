import numpy as np
import pandas as pd
import util.nlp_features as nf
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack, csr_matrix
from sklearn import model_selection
from model.run_lightGBM import run_lightGBM, make_submission
import gc
import lightgbm as lgb

# class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../data/train.csv').fillna(' ')
test = pd.read_csv('../data/test.csv').fillna(' ')
dat = pd.concat([train, test], axis=0)

dat = nf.text_mining_v1(dat, 18)

dat.to_csv("../data/svd_title_desc_18comp.csv")




### Other text components
dat['all_txt'] = dat.region + ' ' + dat.city + ' ' + dat.parent_category_name + ' ' + dat.category_name + ' ' \
            + dat.param_1 + ' ' + dat.param_2 + ' ' + dat.param_3 + ' ' + dat.title + ' ' + dat.description
dat['param_all'] = dat.param_1 + ' ' + dat.param_2 + ' ' + dat.param_3

col = 'all_txt'
n_comp=18

dat_svd = nf.text_mining_v1(dat, 18)
dat_svd.to_csv("../data/svd_title_desc_18comp.csv")
