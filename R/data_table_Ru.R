rm(list=ls());gc()
Sys.setlocale(,"russian")
library(data.table)
library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)
set.seed(0)


# Load data ---------------------------------------------------------------
tr = read_csv('../Avito-Demand-Prediction-Challenge/data/train.csv')
setDT(tr)
te = read_csv('../Avito-Demand-Prediction-Challenge/data/test.csv')
setDT(te)

tri <- 1:nrow(tr)
# tr[, train_flag := 1]
# te[, train_flag := 0]
y <- tr$deal_probability
tr$deal_probability = NULL
dat = rbind(tr, te)



# Feature engineering -----------------------------------------------------
dat[,price := log1p(price)]
dat[,txt := paste(city, param_1, param_2, param_3, title, description, sep = " ")]
dat[, mon := month(activation_date)]
dat[, mday := mday(activation_date)]
dat[, week := week(activation_date)]
dat[, wday := wday(activation_date)]
col_to_drop = c('item_id', 'user_id', 'city', 'param_1', 'param_2', 'param_3', 
                'title', 'description', 'activation_date', 'image')
dat = dat[, !col_to_drop, with = F]
dat[, price := ifelse(is.na(price), -1, price)]
dat[, image_top_1 := ifelse(is.na(image_top_1), -1, image_top_1)]

gc()


# NLP ---------------------------------------------------------------------
cat("Parsing text...\n")
it_dt <- dat %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()
# <itoken>
#   Inherits from: <iterator>
#   Public:
#   chunk_size: 201187
# clone: function (deep = FALSE) 
#   counter: 0
# ids: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 ...
# initialize: function (iterable, ids = NULL, n_chunks = 10, progress_ = interactive(), 
#                       is_complete: active binding
#                       iterable: list
#                       length: active binding
#                       nextElem: function () 
#                         preprocessor: list
#                       progress: TRUE
#                       progressbar: txtProgressBar
#                       tokenizer: list
                      
vect_dt = create_vocabulary(it_dt, ngram = c(1, 3), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.3, vocab_term_max = 5500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it_dt, vect_dt) %>% 
  fit_transform(m_tfidf)

gc()


# Split into Train & Test -------------------------------------------------
cat("Preparing data...\n")
X <- dat %>% 
  select(-txt) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf)

# Save dataset ------------------------------------------------------------
# saveRDS(X, file = './data/tfidf_1_3grams_3_03_50000.rds')

gc()

train = X[tri,]
test = X[-tri,]
gc()


# Modeling ----------------------------------------------------------------
dtest <- xgb.DMatrix(data = test)
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = train[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = train[-tri, ], label = y[-tri])
cols <- colnames(train)

gc()

cat("Training model...\n")
p <- list(objective = "reg:logistic",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 8,
          eta = 0.05,
          max_depth = 17,
          min_child_weight = 3,
          gamma = 0,
          subsample = 0.8,
          colsample_bytree = 0.7,
          alpha = 0,
          lambda = 0,
          nrounds = 2000)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 50)
# [1500]	val-rmse:0.223527 

xgb.importance(cols, model=m_xgb) %>%
  xgb.plot.importance(top_n = 15)


# Submissions -------------------------------------------------------------
cat("Creating submission file...\n")
read_csv("./data/sample_submission.csv") %>%  
  mutate(deal_probability = predict(m_xgb, dtest)) %>%
  write_csv(paste0("./submissions/xgb_tfidf_dt_", round(m_xgb$best_score, 5), ".csv"))



# [1]	val-rmse:0.428720 
# Will train until val_rmse hasn't improved in 50 rounds.
# [51]	val-rmse:0.227506 
# [101]	val-rmse:0.223314 
# [151]	val-rmse:0.222583 
# [201]	val-rmse:0.222073 
# [251]	val-rmse:0.221790 
# [301]	val-rmse:0.221554 
# [351]	val-rmse:0.221388 
# [401]	val-rmse:0.221245 
# [451]	val-rmse:0.221144 
# [501]	val-rmse:0.221044 
# [551]	val-rmse:0.220970 
# [601]	val-rmse:0.220899 
# [651]	val-rmse:0.220846 
# [701]	val-rmse:0.220803 
# [751]	val-rmse:0.220746 
# [801]	val-rmse:0.220694 
# [851]	val-rmse:0.220672 
# [901]	val-rmse:0.220631 
# [951]	val-rmse:0.220586 
# [1001]	val-rmse:0.220562 
# [1051]	val-rmse:0.220537 
# [1101]	val-rmse:0.220499 
# [1151]	val-rmse:0.220479 
# [1201]	val-rmse:0.220452 
# [1251]	val-rmse:0.220440 
# [1301]	val-rmse:0.220429 
# [1351]	val-rmse:0.220426 
# Stopping. Best iteration:
# [1338]	val-rmse:0.220419



# [1]	val-rmse:0.428652 
# Will train until val_rmse hasn't improved in 50 rounds.
# 
# [51]	val-rmse:0.226957 
# [101]	val-rmse:0.222981 
# [151]	val-rmse:0.222216 
# [201]	val-rmse:0.221812 
# [251]	val-rmse:0.221489 
# [301]	val-rmse:0.221288 
# [351]	val-rmse:0.221084 
# [401]	val-rmse:0.220944 
# [451]	val-rmse:0.220841 
# [501]	val-rmse:0.220740 
# [551]	val-rmse:0.220684 
# [601]	val-rmse:0.220609 
# [651]	val-rmse:0.220534 
# [701]	val-rmse:0.220488 
# [751]	val-rmse:0.220444 
# [801]	val-rmse:0.220401 
# [851]	val-rmse:0.220362 
# [901]	val-rmse:0.220331 
# [951]	val-rmse:0.220281 
# [1001]	val-rmse:0.220254 
# [1051]	val-rmse:0.220238 
# [1101]	val-rmse:0.220221 
# [1151]	val-rmse:0.220206 
# [1201]	val-rmse:0.220186 
# [1251]	val-rmse:0.220168 
# [1301]	val-rmse:0.220149 
# [1351]	val-rmse:0.220119 
# [1401]	val-rmse:0.220101 
# [1451]	val-rmse:0.220093 
# Stopping. Best iteration:
# [1418]	val-rmse:0.220091