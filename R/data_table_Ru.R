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


# Load data ---------------------------------------------------------------
tr = read_csv('../Avito-Demand-Prediction-Challenge/data/train.csv')
setDT(tr)
te = read_csv('../Avito-Demand-Prediction-Challenge/data/train.csv')
setDT(te)
tr[, train_flag := 1]
te[, train_flag := 0]
y <- tr$deal_probability
dat = rbind(tr, te)



# Feature engineering -----------------------------------------------------
dat[,price := log(price)]
dat[,txt := paste(region, city, param_1, param_2, param_3, title, description, sep = " ")]
dat[, mon := month(activation_date)]
dat[, mday := mday(activation_date)]
dat[, week := week(activation_date)]
dat[, wday := wday(activation_date)]
dat[, price := ifelse(is.na(price), -1, price)]
dat[, image_top_1 := ifelse(is.na(image_top_1), -1, image_top_1)]

gc()


# NLP ---------------------------------------------------------------------
cat("Parsing text...\n")
dat[, txt := str_to_lower(txt)]
dat[, txt := str_replace_all(txt, "[^[:alpha:]]", " ")]
dat[, txt := str_replace_all(txt, "\\s+", " ")]

token = tokenize_word_stems(dat$txt, language = "russian", stopwords = NULL) %>%
  itoken()
vect = create_vocabulary(token, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.3, vocab_term_max = 4000) %>% 
  vocab_vectorizer()
# Number of docs: 3006848 
# 159 stopwords: и, в, во, не, что, он ... 
# ngram_min = 1; ngram_max = 1 
# Vocabulary: 
#   term term_count doc_count
# 1:  состоян     896316    858762
# 2:     прод     857046    721518
# 3:    одежд     846920    694874
# 4:      нов     710082    569656
# 5:      кра     688936    681378
# ---                              
#   3996:  каневск       2080      1986
# 3997: трансфер       2080      1588
# 3998:    прайс       2080      1988
# 3999:     line       2078      1566
# 4000:  premium       2078      1622

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(token, vect) %>% 
  fit_transform(m_tfidf)

gc()


# Split into Train & Test -------------------------------------------------
cat("Preparing data...\n")
tr_idx = 1:nrow(dat[train_flag==1])
col_to_drop = c('item_id', 'user_id', 'city', 'param_1', 'param_2', 'param_3', 'title', 'description', 'activation_date', 'image', 'txt', 'deal_probability')
dat.sparse = sparse.model.matrix(~ -1 + ., dat[, !col_to_drop, with = F], -1)
dat_all = cbind(dat.sparse, tfidf)

gc()

train = dat_all[tr_idx,]
test = dat_all[-tr_idx,]
gc()


# Modeling ----------------------------------------------------------------
dtest <- xgb.DMatrix(data = test)
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = train[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = train[-tri, ], label = y[-tri])
cols <- colnames(train)

gc()

#---------------------------
cat("Training model...\n")
p <- list(objective = "reg:logistic",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 8,
          eta = 0.05,
          max_depth = 11,
          min_child_weight = 1,
          gamma = 0,
          subsample = 0.7,
          colsample_bytree = 0.7,
          alpha = 0,
          lambda = 0,
          nrounds = 4000)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 50)

xgb.importance(cols, model=m_xgb) %>%
  xgb.plot.importance(top_n = 15)

#---------------------------
cat("Creating submission file...\n")
read_csv("../input/sample_submission.csv") %>%  
  mutate(deal_probability = predict(m_xgb, dtest)) %>%
  write_csv(paste0("xgb_tfidf", round(m_xgb$best_score, 5), ".csv"))