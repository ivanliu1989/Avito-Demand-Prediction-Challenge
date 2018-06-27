library(data.table)
library(tidyverse)


pred1 = fread("./submissions/fnl/best_public_blend_0.2204.csv")
pred2 = fread("./submissions/fnl/i_am_back_kernal_blend_22.csv")
pred3 = fread("./submissions/fnl/lgsub_blend.csv") # 0.2228

par(mfcol = c(2,2))
plot(pred1$deal_probability, pred2$deal_probability)
plot(pred1$deal_probability, pred3$deal_probability)
plot(pred2$deal_probability, pred3$deal_probability)


identical(pred1$item_id, pred2$item_id)
identical(pred1$item_id, pred3$item_id)

submit = pred1
submit$deal_probability = 0.5 * (0.6 * pred1$deal_probability + 0.4 * pred2$deal_probability) + 0.5 * pred3$deal_probability


write.csv(submit, file = './submissions/fnl/blend_433_123.csv', row.names = F)





pred1 = fread("./submissions/fnl/lgsub_0.2213.csv") # 0.2228
pred2 = fread("./submissions/fnl/lgsub_1.csv") # 0.2228
pred3 = fread("./submissions/fnl/lgsub_2.csv") # 0.2228
pred4 = fread("./submissions/fnl/lgsub_3.csv") # 0.2228
pred5 = fread("./submissions/fnl/lgsub_4.csv") # 0.2228
pred6 = fread("./submissions/fnl/lgsub_5.csv") # 0.2228

par(mfcol = c(2,3))
plot(pred1$deal_probability, pred2$deal_probability)
plot(pred1$deal_probability, pred3$deal_probability)
plot(pred1$deal_probability, pred4$deal_probability)
plot(pred1$deal_probability, pred5$deal_probability)
plot(pred1$deal_probability, pred6$deal_probability)

submit = pred1
submit$deal_probability = pred1$deal_probability + pred2$deal_probability + pred3$deal_probability + pred4$deal_probability + pred5$deal_probability + pred6$deal_probability
submit$deal_probability = submit$deal_probability / 6
plot(pred1$deal_probability, submit$deal_probability)
write.csv(submit, file = './submissions/fnl/lgsub_blend.csv', row.names = F)

