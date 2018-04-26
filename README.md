# Avito-Demand-Prediction-Challenge

### Setup Env
- conda create -n avito pip python=3.6
- pip install tensorflow-gpu


### Feature Engineering
- City/Category/UseType/Activation_date Ads counts / distribution / percentage
- City/Category/UseType/Activation_date Based Deal Probability Distribution
- Price percentile by City/Category/UseType/Activation_date
- Activation_date -> Weekdays, Weekends, Holidays
- Users shopping behaviour -> overlapping users, shopped categories, frequency, monetary, recency etc.
- Ads title/description text mining -> TF-IDF, top SVD components
- Target encoding: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

### Inferences:
- So the dates are not different between train and test sets. So we need to be careful while doing our validation. May be time based validation is a good option.
- We are given two weeks data for training (March 15 to March 28) and one week data for testing (April 12 to April 18, 2017).
- There is a gap of two weeks in between training and testing data.
- We can probably use weekday as a feature since all the days are present in both train and test sets.
- 68k users are overlapping between test and train
- 64k ads titles are common to train and test
- The top SVD components capture quite an amount of variation in the data. So this might be helpful features in our modeling process.