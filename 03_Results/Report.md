# Reusult's report
## Berkeley Hass Coding Challenge
### Ulises Quevedo - June 2024

The following report documents findings from the coding challenge assigned. The challenge consisted on training a **binary classifier** to categorize tweets as either **poisitve or negative** based on the tweets text. The tweets Entity was also available, although it was not used for reasons outlined below. The report is structured as follows:

1. Executive summary
2. Data pre-processing overview (`data_prep.py`)
3. Modelling overview (`modelling.py`)
4. Results + Other questions
5. Possible further work


# 1. Executive summary
- We **use only tweets with positive / negative sentiment**, dropping those with Irrelevant or Neutral label
- We preprocess tweets text with a standard pipeline: **lowercasing, tokenization by words, removal of punctuation and stopwords**
- We create **"weight" features** for each tweet, based on the frequency of words in positive/negative tweets. We repeat this with bigrams, which prove less useful but allow us to find patterns such as names ("Assassins Creed", "Red Dead"). We add **length of tweet** as an additional feature.
- We package all preprocessing into a scikit-learn pipeline to make it modular and easy to modify.
- We train two models: a logistic regression based only on weight features and an XGBoost classifier based on weight features + length. We fine tune both using optuna and a validation dataset, seeking to maximize the ROC AUC metric. 
- We present a report for different classification method and explore the effect of vocabulary size on models' performance. 


# 2. Data preprocessing

