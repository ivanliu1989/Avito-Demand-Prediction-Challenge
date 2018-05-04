library(data.table)
library(tidyverse)
train_dat <- read_csv('./data/train.csv')
# setDT(train_dat)
# train_dat
# train_dat <- fread('./data/train.csv', encoding = 'UTF-8')

train_dat$region


pred1 = fread("./submissions/xgb_tfidf_dt_0.22042.csv")
pred2 = fread("./submissions/blend/blend_tfidf_baseline.csv")
pred3 = fread("./submissions/xgb_tfidf0.22017.csv")

par(mfcol = c(2,2))
plot(pred1$deal_probability, pred2$deal_probability)
plot(pred1$deal_probability, pred3$deal_probability)
plot(pred3$deal_probability, pred2$deal_probability)
