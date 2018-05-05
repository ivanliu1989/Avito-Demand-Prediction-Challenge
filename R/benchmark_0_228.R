library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)
set.seed(0)

#---------------------------
cat("Loading data...\n")
tr <- read_csv("./data/train.csv") 
te <- read_csv("./data/test.csv")

#---------------------------
cat("Preprocessing...\n")
tri <- 1:nrow(tr)
y <- tr$deal_probability

tr_te <- tr %>% 
  select(-deal_probability) %>% 
  bind_rows(te) %>% 
  mutate(category_name = as_factor(category_name),
         parent_category_name = as_factor(parent_category_name),
         region = as_factor(region),
         user_type = as_factor(user_type),
         price = log1p(price),
         txt = paste(city, param_1, param_2, param_3, title, description, sep = " "),
         mon = month(activation_date),
         mday = mday(activation_date),
         week = week(activation_date),
         wday = wday(activation_date)) %>% 
  select(-item_id, -user_id, -city, -param_1, -param_2, -param_3,
         -title, -description, -activation_date, -image) %>% 
  replace_na(list(image_top_1 = -1, price = -1)) %T>% 
  glimpse()

rm(tr, te); gc()

#---------------------------
cat("Parsing text...\n")
it <- tr_te %$%
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
          
vect <- create_vocabulary(it, ngram = c(1, 3), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.3, vocab_term_max = 5500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>%
  fit_transform(m_tfidf)



rm(it, vect, m_tfidf); gc()

#---------------------------
cat("Preparing data...\n")
X <- tr_te %>% 
  select(-txt) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf)

rm(tr_te, tfidf); gc()

dtest <- xgb.DMatrix(data = X[-tri, ])
X <- X[tri, ]; gc()
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = X[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = X[-tri, ], label = y[-tri])
cols <- colnames(X)

rm(X, y, tri); gc()

#---------------------------
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

xgb.importance(cols, model=m_xgb) %>%
  xgb.plot.importance(top_n = 15)

#---------------------------
cat("Creating submission file...\n")
read_csv("./data/sample_submission.csv") %>%  
  mutate(deal_probability = predict(m_xgb, dtest)) %>%
  write_csv(paste0("xgb_tfidf", round(m_xgb$best_score, 5), ".csv"))


# [1]	val-rmse:0.428802 
# Will train until val_rmse hasn't improved in 50 rounds.
# [51]	val-rmse:0.228356 
# [101]	val-rmse:0.223992 
# [151]	val-rmse:0.222848 
# [201]	val-rmse:0.222272 
# [251]	val-rmse:0.221857 
# [301]	val-rmse:0.221604 
# [351]	val-rmse:0.221384 
# [401]	val-rmse:0.221216 
# [451]	val-rmse:0.221102 
# [501]	val-rmse:0.220997 
# [551]	val-rmse:0.220897 
# [601]	val-rmse:0.220807 
# [651]	val-rmse:0.220732 
# [701]	val-rmse:0.220680 
# [751]	val-rmse:0.220616 
# [801]	val-rmse:0.220547 
# [851]	val-rmse:0.220515 
# [901]	val-rmse:0.220447 
# [951]	val-rmse:0.220405 
# [1001]	val-rmse:0.220368 
# [1051]	val-rmse:0.220335 
# [1101]	val-rmse:0.220310 
# [1151]	val-rmse:0.220282 
# [1201]	val-rmse:0.220262 
# [1251]	val-rmse:0.220233 
# [1301]	val-rmse:0.220219 
# [1351]	val-rmse:0.220186 
# [1401]	val-rmse:0.220182 
# [1451]	val-rmse:0.220176 
# Stopping. Best iteration:
# [1442]	val-rmse:0.220172