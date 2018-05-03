import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack, csr_matrix

# class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../data/train.csv').fillna(' ')
test = pd.read_csv('../data/test.csv').fillna(' ')

train_text = train.region + ' ' + train.city + ' ' + train.parent_category_name + ' ' + train.category_name + ' ' \
             + train.param_1 + ' ' + train.param_2 + ' ' + train.param_3 + ' ' + train.title + ' ' + train.description
test_text = test.region + ' ' + test.city + ' ' + test.parent_category_name + ' ' + test.category_name + ' ' \
            + test.param_1 + ' ' + test.param_2 + ' ' + test.param_3 + ' ' + test.title + ' ' + test.description
all_text = pd.concat([train_text, test_text])

stopWords = stopwords.words('russian')
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    # strip_accents='unicode',
    analyzer='word',
    # token_pattern=r'\w{1,}',
    stop_words=stopWords,
    ngram_range=(1, 1),
    max_features=10000,
    norm='l2',
    min_df=3,
    max_df=0.3)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    # strip_accents='unicode',
    analyzer='char',
    stop_words=stopWords,
    ngram_range=(2, 6),
    max_features=50000,
    norm='l2',
    min_df=3,
    max_df=0.3)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

# Save train sparse
np.save('train_words_and_ngrams_feature',train_features.data)
np.save('train_words_and_ngrams_indices',train_features.indices)
np.save('train_words_and_ngrams_indptr',train_features.indptr)
# Save test sparse
np.save('test_words_and_ngrams_feature',test_features.data)
np.save('test_words_and_ngrams_indices',test_features.indices)
np.save('test_words_and_ngrams_indptr',test_features.indptr)

# Load train sparse
train_features = np.load('train_words_and_ngrams_feature.npy')
indices = np.load('train_words_and_ngrams_indices.npy')
indptr = np.load('train_words_and_ngrams_indptr.npy')
train_features = csr_matrix((train_features,indices,indptr))
train_features.toarray()
# Load test sparse
test_features = np.load('test_words_and_ngrams_feature.npy')
indices = np.load('test_words_and_ngrams_indices.npy')
indptr = np.load('test_words_and_ngrams_indptr.npy')
test_features = csr_matrix((train_features,indices,indptr))
test_features.toarray()


from sklearn import model_selection
from model.run_lightGBM import run_lightGBM, make_submission
import gc
import lightgbm as lgb
# get model datasets
Y = train.deal_probability
train_X, val_X, train_y, val_y = model_selection.train_test_split(train_features,Y,test_size=0.1, random_state=19)
gc.collect()
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

pred_test_y, model, evals_result, cv_results = run_lightGBM(train_X, train_y, val_X, val_y, test_features,
                                                            params=params, early_stop=100, rounds=5000)
### 5. make submission
res = make_submission(test.item_id, pred_test_y, filename='v0_0_1_2_rmse0_225142_sd0_000170338')



#
# scores = []
# submission = pd.DataFrame.from_dict({'id': test['id']})
# for class_name in class_names:
#     train_target = train[class_name]
#     classifier = LogisticRegression(C=0.1, solver='sag')
#
#     cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
#     scores.append(cv_score)
#     print('CV score for class {} is {}'.format(class_name, cv_score))
#
#     classifier.fit(train_features, train_target)
#     submission[class_name] = classifier.predict_proba(test_features)[:, 1]
#
# print('Total CV score is {}'.format(np.mean(scores)))
#
# submission.to_csv('submission.csv', index=False)
