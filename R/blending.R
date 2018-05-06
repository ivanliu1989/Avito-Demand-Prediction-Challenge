library(data.table)
library(tidyverse)


pred1 = fread("./submissions/baseline_lgb.csv")
pred2 = fread("./submissions/xgb_tfidf0.22012_lb0.2255.csv")
pred3 = fread("./submissions/xgb_tfidf0.2196_lb0.2254.csv")
pred4 = fread("./submissions/xgb_tfidf0.22017.csv")
pred5 = fread("./submissions/xgb_tfidf_dt_0.22042.csv")
pred6 = fread("./submissions/blend/blend_tfidf_baseline.csv")

pred7 = fread("./submissions/xgb_tfidf_dt_0.21941_0.2251.csv")
pred8 = fread("./submissions/xgb_tfidf_dt_0.21946_v0.4.1.csv")
pred9 = fread("./submissions/xgb_tfidf_dt_0.21952_v0.4.2.csv")
pred10 = fread("./submissions/xgb_tfidf_dt_0.21946_v0.4.3.csv")
pred11 = fread("./submissions/xgb_tfidf_dt_0.21966_0.2246.csv")
pred12 = fread("./submissions/blend/blend_tfidf_baseline3.csv")

par(mfcol = c(2,2))
plot(pred12$deal_probability, pred11$deal_probability)
plot(pred12$deal_probability, pred7$deal_probability)
plot(pred10$deal_probability, pred11$deal_probability)
plot(pred10$deal_probability, pred8$deal_probability)

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
