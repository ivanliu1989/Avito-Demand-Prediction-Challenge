# Image quality
# WordBatch https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47295
# Mercari https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50256
# https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s

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
rm(tr, te)


# Feature engineering -----------------------------------------------------
dat[, is_desc := ifelse(is.na(description), 0, 1)]
dat[, is_price := ifelse(is.na(price), 0, 1)]
dat[, is_img := ifelse(is.na(image), 0, 1)]
dat[, is_p1 := ifelse(is.na(param_1), 0, 1)]
dat[, is_p2 := ifelse(is.na(param_2), 0, 1)]
dat[, is_p3 := ifelse(is.na(param_3), 0, 1)]

missing_val = 'отсутствует'
dat[['description']][is.na(dat[['description']])] = missing_val

# hist(log1p(dat$price), 100)
dat[, price_log := log1p(price)]
dat[, itm_seq_log := log1p(item_seq_number)]

dat[, mon := month(activation_date)]
dat[, mday := mday(activation_date)]
dat[, week := week(activation_date)]
dat[, wday := weekdays(activation_date)]

# New
dat[, wend := ifelse(wday %in% c('Sunday', 'Saturday'), 1, 0)]
dat[, title_len := nchar(title)]
dat[, desc_len := nchar(description)]
dat[, desc_len := ifelse(is.na(desc_len), 0, desc_len)]
dat[, title_wc := lengths(gregexpr("\\W+", title)) + 1]
dat[, desc_wc := lengths(gregexpr("\\W+", description)) + 1]

dat[, title_punc := str_count(title, '[:punct:]')]
dat[, title_punc_suc := str_count(title, '\\b[:punct:]{2,}\\b')]
dat[, title_num := str_count(title, '[0-9]')]
dat[, title_num_suc := str_count(title, '\\b[0-9]{2,}\\b')]
dat[, title_upper := str_count(title, '[A-Z]')]
dat[, title_upper_suc := str_count(title, "\\b[A-Z]{2,}\\b")]
dat[, title_ascii := str_count(title, '[:ascii:]')]
dat[, title_ascii_suc := str_count(title, "\\b[:ascii:]{2,}\\b")]
dat[, title_space := str_count(title, '[:blank:]')]

dat[, title_punc_p := title_punc / title_len]
dat[, title_punc_suc_p := title_punc_suc / title_len]
dat[, title_num_p := title_num / title_len]
dat[, title_num_suc_p := title_num_suc / title_len]
dat[, title_upper_p := title_upper / title_len]
dat[, title_upper_suc_p := title_upper_suc / title_len]
dat[, title_ascii_p := title_ascii / title_len]
dat[, title_ascii_suc_p := title_ascii_suc / title_len]
dat[, title_space_p := title_space / title_len]

dat[, desc_punc := str_count(description, '[:punct:]')]
dat[, desc_punc_suc := str_count(description, '\\b[:punct:]{2,}\\b')]
dat[, desc_num := str_count(description, '[0-9]')]
dat[, desc_num_suc := str_count(description, '\\b[0-9]{2,}\\b')]
dat[, desc_upper := str_count(description, '[A-Z]')]
dat[, desc_upper_suc := str_count(description, "\\b[A-Z]{2,}\\b")]
dat[, desc_ascii := str_count(description, '[:ascii:]')]
dat[, desc_ascii_suc := str_count(description, "\\b[:ascii:]{2,}\\b")]
dat[, desc_space := str_count(description, '[:blank:]')]

dat[, desc_punc_p := desc_punc / desc_len]
dat[, desc_punc_suc_p := desc_punc_suc / desc_len]
dat[, desc_num_p := desc_num / desc_len]
dat[, desc_num_suc_p := desc_num_suc / desc_len]
dat[, desc_upper_p := desc_upper / desc_len]
dat[, desc_upper_suc_p := desc_upper_suc / desc_len]
dat[, desc_ascii_p := desc_ascii / desc_len]
dat[, desc_ascii_suc_p := desc_ascii_suc / desc_len]
dat[, desc_space_p := desc_space / desc_len]
# stopword_count
dat[, title_stops := str_count(title, stopwords("ru"))]
dat[, title_stops := ifelse(is.na(title_stops), 0, title_stops)]
dat[, title_stops_p := title_stops / title_wc]
dat[, desc_stops := str_count(description, stopwords("ru"))]
dat[, desc_stops := ifelse(is.na(desc_stops), 0, desc_stops)]
dat[, desc_stops_p := desc_stops / desc_wc]
# new lines
dat[, title_nline := str_count(title, '\n')]
dat[, title_nline := ifelse(is.na(title_nline), 0, title_nline)]
dat[, desc_nline := str_count(description, '\n')]
dat[, desc_nline := ifelse(is.na(desc_nline), 0, desc_nline)]



# Shopping behaviour
dat[, user_act_cnt := .N, by = user_id] # user's total activation - loyalty
dat[, user_act_cnt_log := log1p(user_act_cnt)]

dat[, user_act_pcat_cnt := .N, by = .(user_id, parent_category_name)] # user's frequency by category
dat[, user_act_cat_cnt := .N, by = .(user_id, category_name)] # user's frequency by category
dat[, pcat_pref := user_act_pcat_cnt / user_act_cnt]
dat[, cat_pref := user_act_cat_cnt / user_act_cnt]

dat[, user_act_date := length(unique(activation_date)), by = user_id] # user's activation dates - frequency

dat[, user_act_mon := mean(price, na.rm = T), by = user_id] # user's average price - monetary
dat[, user_act_mon_sd := sd(price, na.rm = T), by = user_id]
dat[, user_act_mon_log := log1p(user_act_mon)] 
dat[, user_act_mon_ratio := price/user_act_mon] # monetary ratio
dat[, user_act_mon_ratio_log := log1p(user_act_mon_ratio)] 

dat[, user_act_pcat_mon := mean(price, na.rm = T), by = .(user_id, parent_category_name)] # user's average price by category
dat[, user_act_mon_pcat_sd := sd(price, na.rm = T), by = .(user_id, parent_category_name)]
dat[, user_act_mon_pcat_log := log1p(user_act_pcat_mon)] 
dat[, user_act_mon_pcat_ratio := price/user_act_pcat_mon] # monetary ratio
dat[, user_act_mon_pcat_ratio_log := log1p(user_act_mon_pcat_ratio)] 

dat[, user_act_cat_mon := mean(price, na.rm = T), by = .(user_id, category_name)] # user's average price by category
dat[, user_act_mon_cat_sd := sd(price, na.rm = T), by = .(user_id, category_name)]
dat[, user_act_mon_cat_log := log1p(user_act_cat_mon)] 
dat[, user_act_mon_cat_ratio := price/user_act_cat_mon] # monetary ratio
dat[, user_act_mon_cat_ratio_log := log1p(user_act_mon_cat_ratio)] 


# Category/Date - price/image/item_seq_number (percentile)
dat[, cat_cnt := .N, by = .(category_name)]
dat[, pcat_cnt := .N, by = .(parent_category_name)]

dat[, cat_price := mean(price, na.rm = T), by = .(category_name)]
dat[, cat_price_sd := sd(price, na.rm = T), by = .(category_name)]
dat[, cat_price_log := log1p(cat_price)]
dat[, cat_price_ratio := price/cat_price]

dat[, pcat_price := mean(price, na.rm = T), by = .(parent_category_name)]
dat[, pcat_price_sd := sd(price, na.rm = T), by = .(parent_category_name)]
dat[, pcat_price_log := log1p(pcat_price)]
dat[, pcat_price_ratio := price/pcat_price]

dat[, cat_img := mean(image_top_1, na.rm = T), by = .(category_name)]
dat[, cat_img_sd := sd(image_top_1, na.rm = T), by = .(category_name)]
dat[, cat_img_log := log1p(cat_img)]
dat[, cat_img_ratio := image_top_1/cat_img]

dat[, pcat_img := mean(image_top_1, na.rm = T), by = .(parent_category_name)]
dat[, pcat_img_sd := sd(image_top_1, na.rm = T), by = .(parent_category_name)]
dat[, pcat_img_log := log1p(pcat_img)]
dat[, pcat_img_ratio := image_top_1/pcat_img]

dat[, cat_seq := mean(item_seq_number, na.rm = T), by = .(category_name)]
dat[, cat_seq_sd := sd(item_seq_number, na.rm = T), by = .(category_name)]
dat[, cat_seq_log := log1p(cat_seq)]
dat[, cat_seq_ratio := item_seq_number/cat_seq]

dat[, pcat_seq := mean(item_seq_number, na.rm = T), by = .(parent_category_name)]
dat[, pcat_seq_sd := sd(item_seq_number, na.rm = T), by = .(parent_category_name)]
dat[, pcat_seq_log := log1p(pcat_seq)]
dat[, pcat_seq_ratio := item_seq_number/pcat_seq]


dat[, cat_date_cnt := .N, by = .(category_name, activation_date)]
dat[, pcat_date_cnt := .N, by = .(parent_category_name, activation_date)]

dat[, cat_date_price := mean(price, na.rm = T), by = .(category_name, activation_date)]
dat[, cat_date_price_sd := sd(price, na.rm = T), by = .(category_name, activation_date)]
dat[, cat_date_price_log := log1p(cat_date_price)]
dat[, cat_date_price_ratio := price/cat_date_price]

dat[, pcat_date_price := mean(price, na.rm = T), by = .(parent_category_name, activation_date)]
dat[, pcat_date_price_sd := sd(price, na.rm = T), by = .(parent_category_name, activation_date)]
dat[, pcat_date_price_log := log1p(pcat_date_price)]
dat[, pcat_date_price_ratio := price/pcat_date_price]

dat[, cat_date_img := mean(image_top_1, na.rm = T), by = .(category_name, activation_date)]
dat[, cat_date_img_sd := sd(image_top_1, na.rm = T), by = .(category_name, activation_date)]
dat[, cat_date_img_log := log1p(cat_date_img)]
dat[, cat_date_img_ratio := image_top_1/cat_date_img]

dat[, pcat_date_img := mean(image_top_1, na.rm = T), by = .(parent_category_name, activation_date)]
dat[, pcat_date_img_sd := sd(image_top_1, na.rm = T), by = .(parent_category_name, activation_date)]
dat[, pcat_date_img_log := log1p(pcat_date_img)]
dat[, pcat_date_img_ratio := image_top_1/pcat_date_img]

dat[, cat_date_seq := mean(item_seq_number, na.rm = T), by = .(category_name, activation_date)]
dat[, cat_date_seq_sd := sd(item_seq_number, na.rm = T), by = .(category_name, activation_date)]
dat[, cat_date_seq_log := log1p(cat_date_seq)]
dat[, cat_date_seq_ratio := item_seq_number/cat_date_seq]

dat[, pcat_date_seq := mean(item_seq_number, na.rm = T), by = .(parent_category_name, activation_date)]
dat[, pcat_date_seq_sd := sd(item_seq_number, na.rm = T), by = .(parent_category_name, activation_date)]
dat[, pcat_date_seq_log := log1p(pcat_date_seq)]
dat[, pcat_date_seq_ratio := item_seq_number/pcat_date_seq]


# City/Region/Date - price/image/item_seq_number (percentile)
dat[, reg_cnt := .N, by = .(region)]

dat[, reg_price := mean(price, na.rm = T), by = .(region)]
dat[, reg_price_sd := sd(price, na.rm = T), by = .(region)]
dat[, reg_price_log := log1p(reg_price)]
dat[, reg_price_ratio := price/reg_price]

dat[, reg_img := mean(image_top_1, na.rm = T), by = .(region)]
dat[, reg_img_sd := sd(image_top_1, na.rm = T), by = .(region)]
dat[, reg_img_log := log1p(reg_img)]
dat[, reg_img_ratio := image_top_1/reg_img]

dat[, reg_seq := mean(item_seq_number, na.rm = T), by = .(region)]
dat[, reg_seq_sd := sd(item_seq_number, na.rm = T), by = .(region)]
dat[, reg_seq_log := log1p(reg_seq)]
dat[, reg_seq_ratio := item_seq_number/reg_seq]


dat[, reg_date_cnt := .N, by = .(region, activation_date)]

dat[, reg_date_price := mean(price, na.rm = T), by = .(region, activation_date)]
dat[, reg_date_price_sd := sd(price, na.rm = T), by = .(region, activation_date)]
dat[, reg_date_price_log := log1p(reg_date_price)]
dat[, reg_date_price_ratio := price/reg_date_price]

dat[, reg_date_img := mean(image_top_1, na.rm = T), by = .(region, activation_date)]
dat[, reg_date_img_sd := sd(image_top_1, na.rm = T), by = .(region, activation_date)]
dat[, reg_date_img_log := log1p(reg_date_img)]
dat[, reg_date_img_ratio := image_top_1/reg_date_img]

dat[, reg_date_seq := mean(item_seq_number, na.rm = T), by = .(region, activation_date)]
dat[, reg_date_seq_sd := sd(item_seq_number, na.rm = T), by = .(region, activation_date)]
dat[, reg_date_seq_log := log1p(reg_date_seq)]
dat[, reg_date_seq_ratio := item_seq_number/reg_date_seq]



# Param/Date - price/image/item_seq_number (percentile)
dat[, p1_cnt := .N, by = .(param_1)]

dat[, p1_price := mean(price, na.rm = T), by = .(param_1)]
dat[, p1_price_sd := sd(price, na.rm = T), by = .(param_1)]
dat[, p1_price_log := log1p(p1_price)]
dat[, p1_price_ratio := price/p1_price]

dat[, p1_img := mean(image_top_1, na.rm = T), by = .(param_1)]
dat[, p1_img_sd := sd(image_top_1, na.rm = T), by = .(param_1)]
dat[, p1_img_log := log1p(p1_img)]
dat[, p1_img_ratio := image_top_1/p1_img]

dat[, p1_seq := mean(item_seq_number, na.rm = T), by = .(param_1)]
dat[, p1_seq_sd := sd(item_seq_number, na.rm = T), by = .(param_1)]
dat[, p1_seq_log := log1p(p1_seq)]
dat[, p1_seq_ratio := item_seq_number/p1_seq]


dat[, p1_date_cnt := .N, by = .(param_1, activation_date)]

dat[, p1_date_price := mean(price, na.rm = T), by = .(param_1, activation_date)]
dat[, p1_date_price_sd := sd(price, na.rm = T), by = .(param_1, activation_date)]
dat[, p1_date_price_log := log1p(p1_date_price)]
dat[, p1_date_price_ratio := price/p1_date_price]

dat[, p1_date_img := mean(image_top_1, na.rm = T), by = .(param_1, activation_date)]
dat[, p1_date_img_sd := sd(image_top_1, na.rm = T), by = .(param_1, activation_date)]
dat[, p1_date_img_log := log1p(p1_date_img)]
dat[, p1_date_img_ratio := image_top_1/p1_date_img]

dat[, p1_date_seq := mean(item_seq_number, na.rm = T), by = .(param_1, activation_date)]
dat[, p1_date_seq_sd := sd(item_seq_number, na.rm = T), by = .(param_1, activation_date)]
dat[, p1_date_seq_log := log1p(p1_date_seq)]
dat[, p1_date_seq_ratio := item_seq_number/p1_date_seq]


dat[, p2_cnt := .N, by = .(param_2)]

dat[, p2_price := mean(price, na.rm = T), by = .(param_2)]
dat[, p2_price_sd := sd(price, na.rm = T), by = .(param_2)]
dat[, p2_price_log := log1p(p2_price)]
dat[, p2_price_ratio := price/p2_price]

dat[, p2_img := mean(image_top_1, na.rm = T), by = .(param_2)]
dat[, p2_img_sd := sd(image_top_1, na.rm = T), by = .(param_2)]
dat[, p2_img_log := log1p(p2_img)]
dat[, p2_img_ratio := image_top_1/p2_img]

dat[, p2_seq := mean(item_seq_number, na.rm = T), by = .(param_2)]
dat[, p2_seq_sd := sd(item_seq_number, na.rm = T), by = .(param_2)]
dat[, p2_seq_log := log1p(p2_seq)]
dat[, p2_seq_ratio := item_seq_number/p2_seq]


dat[, p2_date_cnt := .N, by = .(param_2, activation_date)]

dat[, p2_date_price := mean(price, na.rm = T), by = .(param_2, activation_date)]
dat[, p2_date_price_sd := sd(price, na.rm = T), by = .(param_2, activation_date)]
dat[, p2_date_price_log := log1p(p2_date_price)]
dat[, p2_date_price_ratio := price/p2_date_price]

dat[, p2_date_img := mean(image_top_1, na.rm = T), by = .(param_2, activation_date)]
dat[, p2_date_img_sd := sd(image_top_1, na.rm = T), by = .(param_2, activation_date)]
dat[, p2_date_img_log := log1p(p2_date_img)]
dat[, p2_date_img_ratio := image_top_1/p2_date_img]

dat[, p2_date_seq := mean(item_seq_number, na.rm = T), by = .(param_2, activation_date)]
dat[, p2_date_seq_sd := sd(item_seq_number, na.rm = T), by = .(param_2, activation_date)]
dat[, p2_date_seq_log := log1p(p2_date_seq)]
dat[, p2_date_seq_ratio := item_seq_number/p2_date_seq]


# dat[, txt := paste(city, param_1, param_2, param_3, sep = " ")] # seperate title and description
# dat[, txt := paste(title, description, sep = " ")]
dat[, description := str_replace_all(str_replace_all(str_to_lower(description),"[^[:alpha:]]", " "), "\\s+", " ")]
dat[, title := str_replace_all(str_replace_all(str_to_lower(title),"[^[:alpha:]]", " "), "\\s+", " ")]
dat[, param_1 := str_replace_all(str_replace_all(str_to_lower(param_1),"[^[:alpha:]]", " "), "\\s+", " ")]
dat[, param_2 := str_replace_all(str_replace_all(str_to_lower(param_2),"[^[:alpha:]]", " "), "\\s+", " ")]
dat[, param_3 := str_replace_all(str_replace_all(str_to_lower(param_3),"[^[:alpha:]]", " "), "\\s+", " ")]

dim(dat)
tr = dat[tri]
te = dat[-tri]
write.csv(tr, file = './data/train_bsc_fe_txt_clean.csv', row.names = F)
write.csv(te, file = './data/test_bsc_fe_txt_clean.csv', row.names = F)


### TODO
# Param 3 / City
# Target mean - item_seq_number
# Percentile

# col_to_drop = c('item_id', 'user_id', 'city', 'param_3', # 'param_1', 'param_2', 
#                 'activation_date', 'image')
# dat = dat[, !col_to_drop, with = F]

# Target Mean
# dat[, deal_region_avg := mean(deal_probability, na.rm = T), by = region]
# dat[, deal_region_sd := sd(deal_probability, na.rm = T), by = region]
# dat[, deal_category_avg := mean(deal_probability, na.rm = T), by = category_name]
# dat[, deal_category_sd := sd(deal_probability, na.rm = T), by = category_name]
# dat[, deal_pcategory_avg := mean(deal_probability, na.rm = T), by = parent_category_name]
# dat[, deal_pcategory_sd := sd(deal_probability, na.rm = T), by = parent_category_name]
# dat[, deal_p1_avg := mean(deal_probability, na.rm = T), by = param_1]
# dat[, deal_p1_sd := sd(deal_probability, na.rm = T), by = param_1]
# dat[, deal_p2_avg := mean(deal_probability, na.rm = T), by = param_2]
# dat[, deal_p2_sd := sd(deal_probability, na.rm = T), by = param_2]
# dat[, deal_usrtype_avg := mean(deal_probability, na.rm = T), by = user_type]
# dat[, deal_usrtype_sd := sd(deal_probability, na.rm = T), by = user_type]
# 
# dat$deal_probability = NULL

# Impute
# for (j in seq_len(ncol(dat))){
#   dat[[j]][is.na(dat[[j]])] = -1
# }
# gc()
