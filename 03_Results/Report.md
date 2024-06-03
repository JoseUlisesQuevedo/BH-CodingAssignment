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
- We create **"weight" features** for each tweet, based on the frequency of words in positive/negative tweets. We repeat this with bigrams, which ***prove less useful but allow us to find patterns such as names** ("Assassins Creed", "Red Dead"). We add **length of tweet** as an additional feature.
- We package all preprocessing into a **scikit-learn pipeline** to make it modular and easy to modify.
- We train two models: a logistic regression based only on weight features and an XGBoost classifier based on weight features + length. We **fine tune both using optuna** and a validation dataset, seeking to maximize the ROC AUC metric. 
- We present a report for different classification method and explore the effect of vocabulary size on models' performance. We find **larger vocab** models are better, and that **XGBoost** preforms better than a vanilla logistic regresion. Likewise, we find that **tweet length** turns out to be a very important feature. 


# 2. Data preprocessing

We first preprocess the data. Since we are working with tweets, it is natural to process the text of the tweets using the standard NLP procedure: tokenization, removal of punctuation and removal of stopwords. While stemming and lemmatization are quite standard as well, we do not use them here. 

## Features 

We create 4 features for each tweet: positive and negative word and bigram weights. These are based on a word (bigram) set that counts the N most frequent words (bigrams) in positive and negative tweets. We then substitute each word (bigram) of a tweet with its weight (giving a weight of 0 if not in the set) and add them up, to find a positive/negative weight. The following code snippet shows how the dictionaries were built, and how the weight is assigned


```python
def get_top_N_words(dataframe, N=100, sentiment="Positive", stop_words=stopwords.words('english')):
   
    # Filters tweets to relevant sentiment
    relevant_tweets = dataframe.query("Sentiment == '{}'".format(sentiment))["Content"].dropna()
    # Makes them lowercase
    lower_tweets = relevant_tweets.apply(str.lower)
    # Removes punctuation signs
    no_punctuation = lower_tweets.apply(lambda x: x.translate(str.maketrans('', '', (string.punctuation+"’0123456789"))))
    # Tokenizes
    tokenized_tweets = no_punctuation.apply(word_tokenize)
    # Removes punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    no_stopwords = tokenized_tweets.apply(lambda x: [word for word in x if word not in stop_words])

    corpus = sum(no_stopwords, [])

    word_counts = Counter(corpus)
    top_N_words = word_counts.most_common(N)

    return dict(top_N_words)


def get_word_weight(tweet, dictionary, stop_words=stopwords.words('english')):
    """
    Calculates the weight of a tweet based on a given dictionary of word weights.

    Parameters:
    tweet (str): The tweet to calculate the weight for.
    dictionary (dict): A dictionary containing word weights.
    stop_words (list, optional): A list of stop words to be excluded from the tweet. Defaults to stopwords.words('english').

    Returns:
    float: The weight of the tweet based on the word weights in the dictionary.
    """

    # Filters tweets to relevant sentiment
    tweet = tweet.lower().translate(str.maketrans('', '', (string.punctuation+"’0123456789"))).split()
    tweet = [word for word in tweet if word not in stop_words]

    weight = sum([dictionary[word] if word in dictionary else 0 for word in tweet])

    return weight
```

Additionally, we include the **tweet's length** as a feature. This under the hypothesis that negative tweets might be longer than positive tweets (people tend to hate on things more than praise). To check this, we carry out a quick EDA that shows that negative tweets are in average longer (this difference proves to be statistically significant through a t-test, but we use this only as a motivating factor).

## Pipeline

In order to make our code as modular and clean as possible, we take advantage of scikit-learn's pipeline interface. We create three classes for preprocessing:

1. TweetCleaner: Removes tweets with Irrelevant or Neutral labels, drops na's (tweets with no content) and remaps our label to 0 (Negative) and 1 (Positive)
2. TweetFeatureExtractor: Creates the features outlined above. We implement fit and transform methods, we fit the word/bigram dictionary on the training data and transform the test and validation data with this.
3. DF_Cleaner: Splits the dataset into Features and Labels, keeping only relevant columns

## Splitting

We follow a pretty standard policy for data splitting, since there is no temporal element to care for. Since classes are well balanced (22542 Negative vs 20832 Positive), we use a regular split, but stratify our samples just for good measure. We **save 20% of the data (8698 examples) for testing**, and the remaining 80% is **split again into testing and validation, with a 20% validation set (27886 testing, 6972 validation)**

The processing pipeline ends by creating the datasets and their labels and saving these csv files. 