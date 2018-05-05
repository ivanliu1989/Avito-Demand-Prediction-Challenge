library(data.table)
library(tidyverse)


pred1 = fread("./submissions/baseline_lgb.csv")
pred2 = fread("./submissions/xgb_tfidf0.22012_lb0.2255.csv")
pred3 = fread("./submissions/xgb_tfidf0.2196_lb0.2254.csv")
pred4 = fread("./submissions/xgb_tfidf0.22017.csv")
pred5 = fread("./submissions/xgb_tfidf_dt_0.22042.csv")
pred6 = fread("./submissions/blend/blend_tfidf_baseline.csv")

plot(pred2$deal_probability, pred3$deal_probability)
plot(pred6$deal_probability, pred1$deal_probability)
plot(pred4$deal_probability, pred2$deal_probability)


identical(pred1$item_id, pred6$item_id)

submit = pred2
# submit$deal_probability = (pred2$deal_probability + pred3$deal_probability)/2
submit$deal_probability = 0.15 * pred1$deal_probability + 0.25 * pred2$deal_probability + 0.25 * pred3$deal_probability +
  0.1 * pred4$deal_probability + 0.1 * pred5$deal_probability + 0.15 * pred6$deal_probability


write.csv(submit, file = './submissions/blend/blend_tfidf_baseline3.csv', row.names = F)
