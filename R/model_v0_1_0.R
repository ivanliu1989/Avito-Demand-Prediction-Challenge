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
y <- tr$deal_probability
tr$deal_probability = NULL
dat = rbind(tr, te)



# Feature engineering -----------------------------------------------------
dat[, price := log1p(price)]
# dat[, txt := paste(city, param_1, param_2, param_3, title, description, sep = " ")]
dat[, txt := paste(title, description, sep = " ")]
dat[, mon := month(activation_date)]
dat[, mday := mday(activation_date)]
dat[, week := week(activation_date)]
dat[, wday := weekdays(activation_date)]

# New
dat[, wend := ifelse(wday %in% c('Sunday', 'Saturday'), 1, 0)]
dat[, image_available := ifelse(is.na(image), 1, 0)]
dat[, title_len := nchar(title)]
dat[, desc_len := nchar(description)]
dat[, desc_len := ifelse(is.na(desc_len), 0, desc_len)]
dat[, title_wc := lengths(gregexpr("\\W+", title)) + 1]
dat[, desc_wc := lengths(gregexpr("\\W+", description)) + 1]
# dat[, param := paste(param_1, param_2, param_3, sep = " ")]
# region 28
# city 1752
# param_1 372
# param_2 278
# param_3 1277
# param 2402
# category 47
# parent_category_name 9

col_to_drop = c('item_id', 'user_id', 'city', 'title', # 'param_1', 'param_2', 'param_3', 
                'description', 'activation_date', 'image')
dat = dat[, !col_to_drop, with = F]
dat[, price := ifelse(is.na(price), -1, price)]
dat[, image_top_1 := ifelse(is.na(image_top_1), -1, image_top_1)]

gc()


# NLP ---------------------------------------------------------------------
cat("Parsing text...\n")
dat[, txt := str_to_lower(txt)]
dat[, txt := str_replace_all(txt, "[^[:alpha:]]", " ")]
dat[, txt := str_replace_all(txt, "\\s+", " ")]

it = tokenize_word_stems(dat$txt, language = "russian") %>%
  itoken()
vect = create_vocabulary(it, ngram = c(1, 3), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.3, vocab_term_max = 5500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>% 
  fit_transform(m_tfidf)

gc()


# Split into Train & Test -------------------------------------------------
cat("Preparing data...\n")
dat[, region := ifelse(is.na(region), 'na', region)]
dat[, parent_category_name := ifelse(is.na(parent_category_name), 'na', parent_category_name)]
dat[, category_name := ifelse(is.na(category_name), 'na', category_name)]
dat[, param_1 := ifelse(is.na(param_1), 'na', param_1)]
dat[, param_2 := ifelse(is.na(param_2), 'na', param_2)]
dat[, param_3 := ifelse(is.na(param_3), 'na', param_3)]
dat[, user_type := ifelse(is.na(user_type), 'na', user_type)]
dat[, wday := ifelse(is.na(wday), 'na', wday)]

X = dat[, !c('txt'), with = F] %>% 
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


# [1]	val-rmse:0.428663 
# Will train until val_rmse hasn't improved in 50 rounds.
# 
# [51]	val-rmse:0.227693 
# [101]	val-rmse:0.223965 
# [151]	val-rmse:0.223125 
# [201]	val-rmse:0.222668 
# [251]	val-rmse:0.222375 
# [301]	val-rmse:0.222145 
# [351]	val-rmse:0.221957 
# [401]	val-rmse:0.221808 
# [451]	val-rmse:0.221663 
# [501]	val-rmse:0.221560 
# [551]	val-rmse:0.221464 
# [601]	val-rmse:0.221393 
# [651]	val-rmse:0.221283 
# [701]	val-rmse:0.221234 
# [751]	val-rmse:0.221157 
# [801]	val-rmse:0.221115 
# [851]	val-rmse:0.221090 
# [901]	val-rmse:0.221024 
# [951]	val-rmse:0.220987 
# [1001]	val-rmse:0.220951 
# [1051]	val-rmse:0.220925 
# [1101]	val-rmse:0.220884 
# [1151]	val-rmse:0.220885 
# [1201]	val-rmse:0.220857 
# [1251]	val-rmse:0.220834 
# [1301]	val-rmse:0.220824 
# [1351]	val-rmse:0.220811 
# [1401]	val-rmse:0.220801 
# [1451]	val-rmse:0.220797 
# Stopping. Best iteration:
# [1413]	val-rmse:0.220792
