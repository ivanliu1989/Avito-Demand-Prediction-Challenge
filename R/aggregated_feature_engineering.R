rm(list=ls());gc()
Sys.setlocale(,"russian")
library(data.table)
library(stringr)
library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)
set.seed(0)

tr = read_csv('./data/train.csv')
setDT(tr)
te = read_csv('./data/test.csv')
setDT(te)

tr_active = read_csv('./data/train_active.csv')
setDT(tr_active)
te_active = read_csv('./data/test_active.csv')
setDT(te_active)

tr_period = read_csv('./data/periods_train.csv')
setDT(tr_period)
te_period = read_csv('./data/periods_test.csv')
setDT(te_period)
periods = rbind(tr_period, te_period)
rm(tr_period, te_period)
gc()

all_sample = rbind(tr_active[,.(item_id, user_id, region, city, parent_category_name, category_name, item_seq_number, price, user_type)],
                   te_active[,.(item_id, user_id, region, city, parent_category_name, category_name, item_seq_number, price, user_type)],
                   tr[,.(item_id, user_id, region, city, parent_category_name, category_name, item_seq_number, price, user_type)],
                   te[,.(item_id, user_id, region, city, parent_category_name, category_name, item_seq_number, price, user_type)])
gc()

periods[, days_up := as.numeric(date_to - date_from)]
periods[, days_ads_placed := as.numeric(activation_date - date_from)]

periods[, days_up_sum_itm := sum(days_up, na.rm = T), by = item_id]
periods[, days_placed_avg_itm := -mean(days_ads_placed, na.rm = T), by = item_id]
periods[, times_put_up_itm := .N, by = item_id]

all_sample = merge(all_sample, unique(periods[,.(item_id, times_put_up_itm, days_up_sum_itm, days_placed_avg_itm)]), by = 'item_id', all.x = T)


# Aggregated feature 1
all_sample[, avg_days_up_user := mean(days_up_sum_itm, na.rm = T), by = user_id]
all_sample[, avg_times_up_user := mean(times_put_up_itm, na.rm = T), by = user_id]
all_sample[, avg_days_placed_user := mean(days_placed_avg_itm, na.rm = T), by = user_id]
all_sample[, avg_item_seq_number_active_log := log1p(mean(item_seq_number, na.rm = T)), by = user_id]
all_sample[, avg_price_active_log := log1p(mean(price, na.rm = T)), by = user_id]

all_sample[, sd_days_up_user := sd(days_up_sum_itm, na.rm = T), by = user_id]
all_sample[, sd_times_up_user := sd(times_put_up_itm, na.rm = T), by = user_id]
all_sample[, sd_days_placed_user := sd(days_placed_avg_itm, na.rm = T), by = user_id]
all_sample[, sd_item_seq_number_active := sd(item_seq_number, na.rm = T), by = user_id]
all_sample[, sd_price_active := sd(price, na.rm = T), by = user_id]

all_sample[, sd_days_up_user := ifelse(is.na(sd_days_up_user), 0, sd_days_up_user)]
all_sample[, sd_times_up_user := ifelse(is.na(sd_times_up_user), 0, sd_times_up_user)]
all_sample[, sd_days_placed_user := ifelse(is.na(sd_days_placed_user), 0, sd_days_placed_user)]
all_sample[, sd_item_seq_number_active := ifelse(is.na(sd_item_seq_number_active), 0, sd_item_seq_number_active)]
all_sample[, sd_price_active := ifelse(is.na(sd_price_active), 0, sd_price_active)]

# Aggregated User Distribution
all_sample[, n_user_items_active := length(unique(item_id)), by = user_id]
all_sample[, user_act_cnt_active := .N, by = user_id]
all_sample[, user_act_pcat_cnt_active := .N, by = .(user_id, parent_category_name)] # user's frequency by category
all_sample[, user_act_cat_cnt_active := .N, by = .(user_id, category_name)] # user's frequency by category
all_sample[, pcat_pref_active := user_act_pcat_cnt_active / user_act_cnt_active]
all_sample[, cat_pref_active := user_act_cat_cnt_active / user_act_cnt_active]


aggregated_user_features = unique(all_sample[, .(user_id, category_name, avg_days_up_user, avg_times_up_user, avg_days_placed_user, 
                                                 avg_item_seq_number_active_log, avg_price_active_log,
                      sd_days_up_user, sd_times_up_user, sd_days_placed_user, sd_item_seq_number_active, sd_price_active,
                      n_user_items_active, user_act_cnt_active, user_act_pcat_cnt_active, user_act_cat_cnt_active, 
                      pcat_pref_active, cat_pref_active)])

write.csv(aggregated_user_features, file = './data/aggregated_features_entire.csv', row.names = F, fileEncoding = "UTF-8")
