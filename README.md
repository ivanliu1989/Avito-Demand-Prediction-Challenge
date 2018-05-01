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
- meta feature - binary, regression, multi-classification (prob bins), pca, clustering

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
	- v0.0.1.1 | val: [2960] 0.225065 + 0.000594688 & LB: 0.230 (50% data)
		- tgt_cols = ['deal_probability', 'price', 'image_top_1', 'activation_date_weekend']
		- cate_cols = ['category_name', 'region', 'city', 'param_1', 'param_2', 'parent_category_name', 'user_type', 'activation_date_dayofweek', 'deal_class', 'deal_class_2', ['activation_date_dayofweek', 'region']]
		- measures = ['mean', 'std', 'quantile', 'skew', 'count']
	- v0.0.1.2 | val: 0.224491
		- v0.0.1.2.1 | v0.0.1.1 + label encoder: [2880] 0.224867 + 0.00055142
		- v0.0.1.2.2 | v0.0.1.1 + user_id & item_id target encoding: [260] 0.125014 + 0.000535261
		- v0.0.1.2.3 | v0.0.1.1 + param_3 target encoding: [2060] 0.224751 + 0.000484106
		- v0.0.1.2.4 | v0.0.1.1 + sorted by date: [2880] 0.224812 + 0.000479768
	- v0.0.1.3 | val: 0.224746
		- v0.0.1.3.1 | v0.0.1.2 + impute price by mean: [2620] 0.224746 + 0.000503914
	- v0.0.1.4 | val:
		- v0.0.1.4.1 | v0.0.1.3 + target encoding with noise: 0.226457 + 0.000489155
		- v0.0.1.4.2 | v0.0.1.4.1 + different measures: 0.226524 + 0.000491616
	- v0.0.1.5 | val:
		- v0.0.1.5.1 | v0.0.1.4 + one hot encoding: [1420] 0.226585 + 0.000516941
	- v0.0.1.6 | val:
		- v0.0.1.6.1 | v0.0.1.5 + word count + image_available: 0.226555 + 0.000530452
- v0.0.2
	- v0.0.2.1 | NLP
		- v0.0.2.1.1 | v0.0.1.6 + tfidf(svd - 3): [1920] 0.225142 + 0.000170338
		- v0.0.2.1.2 | v0.0.2.1.1 + tfidf(svd - 6):
		- v0.0.2.1.3 | v0.0.2.1.2 + big text - title+desc+param123:
		- v0.0.2.1.4 | v0.0.2.1.3 + ngrams:
	- v0.0.2.1 | log transform
- v0.0.3
	- v0.0.3.1 | Image
- v0.0.4
	- v0.0.4.1 | active feature
- v0.0.5
	- v0.0.5.1 | Meta features


### Submit
kaggle competitions submit -c avito-demand-prediction -f v0_0_1_2_rmse0_226457_sd0_000489155.csv -m "test with noise tgt mean"