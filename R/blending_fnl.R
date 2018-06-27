library(data.table)
library(tidyverse)


pred1 = fread("./submissions/fnl/best_public_blend_0.2204.csv")
pred2 = fread("./submissions/fnl/i_am_back_kernal_blend_22.csv")
pred3 = fread("./submissions/fnl/user_id_VGG16_cat2vec_sentiment_0.2213.csv") # 0.2228

par(mfcol = c(2,2))
plot(pred1$deal_probability, pred2$deal_probability)
plot(pred1$deal_probability, pred3$deal_probability)
plot(pred2$deal_probability, pred3$deal_probability)


identical(pred1$item_id, pred2$item_id)
identical(pred1$item_id, pred3$item_id)

submit = pred1
submit$deal_probability = 0.5 * pred1$deal_probability + 0.5 * pred3$deal_probability #+ 0.2 * pred3$deal_probability


write.csv(submit, file = './submissions/fnl/blend_55_13.csv', row.names = F)


