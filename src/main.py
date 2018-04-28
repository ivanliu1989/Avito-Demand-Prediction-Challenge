from util import feature_engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
from collections import Counter

dat = pd.read_csv("../data/train.csv", parse_dates=["activation_date"])
print('train data shape: ', train_dat.shape)
gc.collect()

dat.head()

# City/Category/UseType/Activation_date Ads counts / distribution / percentage
Counter(dat.param_3)
Counter(dat.image_top_1)
cols = ['category_name', 'city', 'param_1', 'param_2', 'param_3']