library(translateR)
library(tidyverse)

getGoogleLanguages()

tr <- read_csv("./data/train.csv") 
te <- read_csv("./data/test.csv")


cols = c('title', 'description', 'user_type',  'param_1', 'param_2', 'param_3', 'parent_category_name', 'category_name', 'region', 'city')

for(c in cols){
  print(c)
  te_parent_category <- translate(dataset = te[,c('item_id', 'user_id', 'activation_date', c)],
                                  content.field = c,
                                  google.api.key = "AIzaSyBXOuRkn4tUTOnSTLqn7BwirkB3gcVpYaA",
                                  source.lang = 'ru',
                                  target.lang = 'en')
  write.csv(te_parent_category, file = paste0("./data/test_",c,".csv"), row.names = F)
}

for(c in cols){
  print(c)
  tr_parent_category <- translate(dataset = tr[,c('item_id', 'user_id', 'activation_date', c)],
                                  content.field = c,
                                  google.api.key = "AIzaSyBXOuRkn4tUTOnSTLqn7BwirkB3gcVpYaA",
                                  source.lang = 'ru',
                                  target.lang = 'en')
  write.csv(tr_parent_category, file = paste0("./data/train_",c,".csv"), row.names = F)
}