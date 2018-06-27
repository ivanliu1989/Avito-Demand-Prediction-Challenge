library(data.table)
library(tidyverse)


pred1 = fread("./submissions/fnl/best_public_blend_0.2204.csv")
pred2 = fread("./submissions/fnl/i_am_back_kernal_blend_22.csv")
pred3 = fread("./submissions/fnl/user_id_VGG16_cat2vec_sentiment_0.2213.csv")

par(mfcol = c(2,2))
plot(pred1$deal_probability, pred2$deal_probability)
plot(pred1$deal_probability, pred3$deal_probability)
plot(pred2$deal_probability, pred3$deal_probability)


identical(pred1$item_id, pred6$item_id)

submit = pred2
# submit$deal_probability = (pred2$deal_probability + pred3$deal_probability)/2
submit$deal_probability = 0.15 * pred1$deal_probability + 0.25 * pred2$deal_probability + 0.25 * pred3$deal_probability +
  0.1 * pred4$deal_probability + 0.1 * pred5$deal_probability + 0.15 * pred6$deal_probability


write.csv(submit, file = './submissions/blend/blend_tfidf_baseline3.csv', row.names = F)



# v0.4
submit = pred2
submit$deal_probability = (pred7$deal_probability + pred8$deal_probability + pred9$deal_probability +
                             pred10$deal_probability + pred11$deal_probability + pred12$deal_probability)/6
write.csv(submit, file = './submissions/blend/blend_tfidf_baseline4.csv', row.names = F)

# v0.5
submit = pred12
submit$deal_probability = 0.55 * pred12$deal_probability + 0.1 * pred13$deal_probability + 0.1 * pred14$deal_probability +
  0.25 * pred15$deal_probability
write.csv(submit, file = './submissions/blend/blend_tfidf_baseline5.csv', row.names = F)



# v0.6
library(data.table)
library(tidyverse)


pred1 = fread("./submissions/xgb_tfidf_dt_0.21861.csv")
pred2 = fread("./submissions/xgb_tfidf_dt_binary_0.21853.csv")
plot(pred1$deal_probability, pred2$deal_probability)
submit = pred2
submit$deal_probability = (pred1$deal_probability + pred2$deal_probability)/2
write.csv(submit, file = './submissions/blend/blend_tfidf_bf.csv', row.names = F)





# Kernals -----------------------------------------------------------------
library(data.table)
library(tidyverse)

pred = fread("./submissions/blend/i_am_back.csv")
pred1 = fread("./submissions/blend/i_am_back.csv")
all_submit = list.files("./submissions/kaggle_kernel/", full.names = T)
# all_submit2 = list.files("./submissions/", full.names = T)
# all_submit3 = list.files("./submissions/blend/", full.names = T)
# all_submit = c(all_submit1, all_submit2, all_submit3)
# all_submit = all_submit[grepl('.csv', all_submit)]
for(f in all_submit){
  print(f)
  pred_f = fread(f)
  if(identical(pred$item_id, pred_f$item_id)){
    pred1$deal_probability = pred1$deal_probability + pred_f$deal_probability
  }else{
    print("id wrong!!!")
  }
}
# pred$deal_probability = (pred$deal_probability + (pred1$deal_probability / (length(all_submit) + 1))) / 2
pred1$deal_probability = pred$deal_probability * 0.25 + (pred1$deal_probability / (length(all_submit) + 1)) * 0.75
plot(pred$deal_probability, pred1$deal_probability)
write.csv(pred1, file = './submissions/blend/i_am_back_kernal_blend_22.csv', row.names = F)
