from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


def text_mining_v1(dat, n_comp=3):

    print('NLP - tfidf')
    # Get Russian Stopwords
    stopWords = stopwords.words('russian')

    # Create tfidf matrix for title and description
    tfidf = TfidfVectorizer(max_features=50000, stop_words=stopWords)
    tfidf_title = TfidfVectorizer(max_features=50000, stop_words=stopWords)

    dat['description'] = dat['description'].fillna(' ')
    dat['title'] = dat['title'].fillna(' ')
    tfidf.fit(dat['description'])
    tfidf_title.fit(dat['title'])

    dat_des_tfidf = tfidf.transform(dat['description'])
    dat_title_tfidf = tfidf.transform(dat['title'])

    # Get Key Components for tfidf matrix
    print('NLP - svd')
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(tfidf.transform(dat['description']))

    svd_title = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_title.fit(tfidf.transform(dat['title']))

    dat_svd = pd.DataFrame(svd_obj.transform(dat_des_tfidf))
    dat_svd.columns = ['svd_des_' + str(i + 1) for i in range(n_comp)]
    # dat = pd.concat([dat, dat_svd], axis=1)
    dat = dat.join(dat_svd)

    dat_title_svd = pd.DataFrame(svd_title.transform(dat_title_tfidf))
    dat_title_svd.columns = ['svd_title_' + str(i + 1) for i in range(n_comp)]
    # dat = pd.concat([dat, dat_title_svd], axis=1)
    dat = dat.join(dat_title_svd)

    return dat
