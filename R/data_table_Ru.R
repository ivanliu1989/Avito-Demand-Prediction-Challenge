Sys.setlocale(,"ru_RU")
Sys.setlocale(,"russian")
library(data.table)
library(readr)
dat = read_csv('../Avito-Demand-Prediction-Challenge/data/train.csv')
setDT(dat)
dat
