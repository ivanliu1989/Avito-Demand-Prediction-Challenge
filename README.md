# Avito-Demand-Prediction-Challenge

### Setup Env
- conda create -n avito pip python=3.6
- pip install tensorflow-gpu


### Feature Engineering
- City/Category/UseType/Activation_date Ads counts / distribution / percentage
- City/Category/UseType/Activation_date Based Deal Probability Distribution
- Max/percentile sequential number
- Price percentile by City/Category/UseType/Activation_date
- Activation_date -> Weekdays, Weekends, Holidays
- Users shopping behaviour -> overlapping users, shopped categories, frequency, monetary, recency etc.
- Ads title/description text mining -> TF-IDF, top SVD components
- Target encoding (e.g. category and deal class, price buckets to deal class): https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
- Param 1/2/3
- ngrams
- sentiment score
- top words - possibility
- meta feature - binary, regression, multi-classification, pca, clustering

### Inferences:
- So the dates are not different between train and test sets. So we need to be careful while doing our validation. May be time based validation is a good option.
- We are given two weeks data for training (March 15 to March 28) and one week data for testing (April 12 to April 18, 2017).
- There is a gap of two weeks in between training and testing data.
- We can probably use weekday as a feature since all the days are present in both train and test sets.
- 68k users are overlapping between test and train
- 64k ads titles are common to train and test
- The top SVD components capture quite an amount of variation in the data. So this might be helpful features in our modeling process.
- From the deal probability distribution graph, it is clear that majority of the items have exteremely low deal probability, ie. about 78%, while very few values have the deal probability of 0.7 or larger.
- A very small tower is observed near the deal probability of 1.0, indicating that there are some items in the dataset having very high value of deal probability.

### Steps
- Russian to Email
- Text mining features
- Image Features

### Model
- catboost 
- xgboost
- lightGBM
- Keras
- Regression

### Feature Engineering
- Func: Time Column & ID Column -> Agg, Transform, Lag
- Func: Text Column -> Word Embeding, Freq, PCA, Cluster
- Func: Target Column (e.g. Price, Deal_Prob) + Categorical Columns -> Target Mean, SD, Median

### Score Tracker
- v0.0.0 | val:0.254438 | feat:'item_seq_number', 'price'
- v0.0.1.0 | val:0.235308 | feat:'image_top_1','item_seq_number','price','activation_date_dayofmonth','activation_date_weekend', 'title_len', 'description_len', 'region_deal_probability_mean','region_deal_probability_median', 'city_deal_probability_mean', 'city_deal_probability_median', 'activation_date_dayofweekregion_deal_probability_mean', 'activation_date_dayofweekregion_deal_probability_median','region_price_mean','region_price_median', 'city_price_mean','city_price_median','activation_date_dayofweekregion_price_mean','activation_date_dayofweekregion_price_median'
- v0.0.1.1 | val: 0.225768 (50%) / 0.222917 (80%) & LB: 0.230 (50% data)
	- tgt_cols = ['deal_probability', 'price', 'image_top_1', 'activation_date_weekend']
	- cate_cols = ['category_name', 'region', 'city', 'param_1', 'param_2', 'parent_category_name', 'user_type', 'activation_date_dayofweek', 'deal_class', 'deal_class_2', ['activation_date_dayofweek', 'region']]
	- measures = ['mean', 'std', 'quantile', 'skew', 'count']
- v0.0.1.2 | val: 0.224703
	- v0.0.1.1 + label encoder
- v0.0.1.3 | val:
	- v0.0.1.2 + impute by mean
- v0.0.1.4 | val:
	- v0.0.1.3 + target encoding with noise