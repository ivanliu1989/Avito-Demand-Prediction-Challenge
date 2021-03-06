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
tr = read_csv('./data/train.csv')
setDT(tr)
te = read_csv('./data/test.csv')
setDT(te)

tri <- 1:nrow(tr)
y <- tr$deal_probability
# tr$deal_probability = NULL
te$deal_probability = NA
dat = rbind(tr, te)



# Feature engineering -----------------------------------------------------
dat[, price := log1p(price)]
dat[, txt := paste(city, param_1, param_2, param_3, title, description, sep = " ")]
# dat[, txt := paste(title, description, sep = " ")]
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

col_to_drop = c('item_id', 'user_id', 'city', 'title', 'param_3', # 'param_1', 'param_2', 
                'description', 'activation_date', 'image')
dat = dat[, !col_to_drop, with = F]

# Target Mean
dat[, price_region_avg := mean(price, na.rm = T), by = region]
dat[, price_region_sd := sd(price, na.rm = T), by = region]
dat[, price_region_ratio := price / price_region_avg]
dat[, price_category_avg := mean(price, na.rm = T), by = category_name]
dat[, price_category_sd := sd(price, na.rm = T), by = category_name]
dat[, price_category_ratio := price / price_category_avg]

dat[, image_region_avg := mean(image_top_1, na.rm = T), by = region]
dat[, image_region_sd := sd(image_top_1, na.rm = T), by = region]
dat[, image_region_ratio := image_top_1 / image_region_avg]
dat[, image_category_avg := mean(image_top_1, na.rm = T), by = category_name]
dat[, image_category_sd := sd(image_top_1, na.rm = T), by = category_name]
dat[, image_category_ratio := image_top_1 / image_category_avg]


dat[, deal_region_avg := mean(deal_probability, na.rm = T), by = region]
dat[, deal_region_sd := sd(deal_probability, na.rm = T), by = region]
dat[, deal_category_avg := mean(deal_probability, na.rm = T), by = category_name]
dat[, deal_category_sd := sd(deal_probability, na.rm = T), by = category_name]

dat$deal_probability = NULL

# Impute
dat[, price := ifelse(is.na(price), -1, price)]
dat[, image_top_1 := ifelse(is.na(image_top_1), -1, image_top_1)]

dat[, price_region_avg := ifelse(is.na(price_region_avg), -1, price_region_avg)]
dat[, price_region_sd := ifelse(is.na(price_region_sd), -1, price_region_sd)]
dat[, price_region_ratio := ifelse(is.na(price_region_ratio), -1, price_region_ratio)]
dat[, price_category_avg := ifelse(is.na(price_category_avg), -1, price_category_avg)]
dat[, price_category_sd := ifelse(is.na(price_category_sd), -1, price_category_sd)]
dat[, price_category_ratio := ifelse(is.na(price_category_ratio), -1, price_category_ratio)]
dat[, image_region_avg := ifelse(is.na(image_region_avg), -1, image_region_avg)]
dat[, image_region_sd := ifelse(is.na(image_region_sd), -1, image_region_sd)]
dat[, image_region_ratio := ifelse(is.na(image_region_ratio), -1, image_region_ratio)]
dat[, image_category_avg := ifelse(is.na(image_category_avg), -1, image_category_avg)]
dat[, image_category_sd := ifelse(is.na(image_category_sd), -1, image_category_sd)]
dat[, image_category_ratio := ifelse(is.na(image_category_ratio), -1, image_category_ratio)]
dat[, deal_region_avg := ifelse(is.na(deal_region_avg), -1, deal_region_avg)]
dat[, deal_region_sd := ifelse(is.na(deal_region_sd), -1, deal_region_sd)]
dat[, deal_category_avg := ifelse(is.na(deal_category_avg), -1, deal_category_avg)]
dat[, deal_category_sd := ifelse(is.na(deal_category_sd), -1, deal_category_sd)]

gc()


# NLP ---------------------------------------------------------------------
cat("Parsing text...\n")
it <- dat %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()
vect = create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.4, vocab_term_max = 6500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>% 
  fit_transform(m_tfidf)

gc()

# ### PCA
# tfidf.pca = read_csv('./data/svd_title_desc_18comp.csv')
# library(sparsesvd)
# tfidf.pcov <- sparsesvd(tfidf)#, scores = TRUE, scale = TRUE, center = TRUE)
# library(irlba)
# prcomp_irlba(tfidf, n = 12, retx = TRUE, center = TRUE, scale. = FALSE)
# ### t-SNE
# tsne3d <- tsne(tfidf, initial_config = NULL, k = 3, initial_dims = 30, perplexity = 35, 
#                max_iter = 1000, min_cost = 0, epoch_callback = NULL, whiten = TRUE, epoch=300)
# tsne3d <- cbind(tsne3d,data[,1])


# Split into Train & Test -------------------------------------------------
cat("Preparing data...\n")
dat[, region := ifelse(is.na(region), 'na', region)]
dat[, parent_category_name := ifelse(is.na(parent_category_name), 'na', parent_category_name)]
dat[, category_name := ifelse(is.na(category_name), 'na', category_name)]
dat[, param_1 := ifelse(is.na(param_1), 'na', param_1)]
dat[, param_2 := ifelse(is.na(param_2), 'na', param_2)]
# dat[, param_3 := ifelse(is.na(param_3), 'na', param_3)]
dat[, user_type := ifelse(is.na(user_type), 'na', user_type)]
dat[, wday := ifelse(is.na(wday), 'na', wday)]

X = dat[, !c('txt'), with = F] %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf) #%>%
# cbind(as.matrix(tfidf.pca))



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
          min_child_weight = 1,
          gamma = 0,
          subsample = 0.7,
          colsample_bytree = 0.7,
          alpha = 0,
          lambda = 0,
          nrounds = 4000)


m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 50)
# [1500]	val-rmse:0.223527 

xgb.importance(cols, model=m_xgb) %>%
  xgb.plot.importance(top_n = 15)


# Submissions -------------------------------------------------------------
cat("Creating submission file...\n")
read_csv("./data/sample_submission.csv") %>%  
  mutate(deal_probability = predict(m_xgb, dtest)) %>%
  write_csv(paste0("./submissions/xgb_tfidf_dt_", round(m_xgb$best_score, 5), ".csv"))


# [1]	val-rmse:0.428754 
# Will train until val_rmse hasn't improved in 50 rounds.
# 
# [51]	val-rmse:0.225579 
# [101]	val-rmse:0.222123 
# [151]	val-rmse:0.221461 
# [201]	val-rmse:0.221135 
# [251]	val-rmse:0.220923 
# [301]	val-rmse:0.220743 
# [351]	val-rmse:0.220614 
# [401]	val-rmse:0.220475 
# [451]	val-rmse:0.220360 
# [501]	val-rmse:0.220265 
# [551]	val-rmse:0.220170 
# [601]	val-rmse:0.220086 
# [651]	val-rmse:0.220012 
# [701]	val-rmse:0.219954 
# [751]	val-rmse:0.219878 
# [801]	val-rmse:0.219815 
# [851]	val-rmse:0.219732 
# [901]	val-rmse:0.219700 
# [951]	val-rmse:0.219674 
# [1001]	val-rmse:0.219639 
# [1051]	val-rmse:0.219608 
# [1101]	val-rmse:0.219564 
# [1151]	val-rmse:0.219553 
# [1201]	val-rmse:0.219542 
# [1251]	val-rmse:0.219501 
# [1301]	val-rmse:0.219472 
# [1351]	val-rmse:0.219443 
# [1401]	val-rmse:0.219437 
# [1451]	val-rmse:0.219415 
# Stopping. Best iteration:
# [1433]	val-rmse:0.219412

