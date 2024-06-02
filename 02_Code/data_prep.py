import pandas as pd
from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
import string
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sys

"""
This module contains functions to preprocess the tweets dataset
It generates a number of features, and splits the dataset into training, testing and validation sets

It uses scikitlearn's interface to create a modular pipeline that allows flexibility to add or remove features

The script takes in a (hyper) parameter N from terminal, that determines the vocabulary size for the top N words and bigrams

"""

def get_top_N_words(dataframe, N=100, sentiment="Positive", stop_words=stopwords.words('english')):
    """
    Returns a dictionary with the top N words by frequency after applying preprocessing.

    Args:
        dataframe (pandas.DataFrame): A dataframe with a 'Content' column containing tweets and a 'Sentiment' column.
        N (int, optional): The number of top words to return. Defaults to 100.
        sentiment (str, optional): The sentiment to filter the tweets. Defaults to "Positive".
        stop_words (list, optional): A list of stopwords to remove from the tweets. Defaults to NLTK stopwords.

    Returns:
        dict: A dictionary with the top N words and their frequencies.

    Preprocessing Steps:
        1. Splits sentences into words (basic tokenization).
        2. Removes punctuation and stopwords (using NLTK stopwords corpus).
        3. Joins everything into a corpus and counts frequencies.
        4. Filters top N.
    """
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


def get_top_N_bigrams(dataframe,N=100,sentiment="Positive",stop_words=stopwords.words('english')):

    """
    Retrieves the top N bigrams by frequency from a dataframe containing tweet data.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the tweet data. Expects a 'Content' column with the tweet text and a 'Sentiment' column.
        N (int, optional): The number of top bigrams to retrieve. Defaults to 100.
        sentiment (str, optional): The sentiment to filter the tweets. Defaults to "Positive".
        stop_words (list, optional): The list of stop words to remove from the tweets. Defaults to nltk.corpus.stopwords.words('english').

    Returns:
        dict: A dictionary containing the top N bigrams and their frequencies.
    """

    #Filters tweets to relevant sentiment
    relevant_tweets = dataframe.query("Sentiment == '{}'".format(sentiment))["Content"].dropna()
    #Makes them lowercase 
    lower_tweets = relevant_tweets.apply(str.lower)
    #Removes punctuation signs
    no_punctuation = lower_tweets.apply(lambda x: x.translate(str.maketrans('', '', (string.punctuation+"’0123456789"))))
    #Tokenizes
    stop_words = set(stopwords.words('english'))

    split_up = no_punctuation.apply(lambda x: x.split())
    no_stopwords = split_up.apply(lambda x: [word for word in x if word not in stop_words])


    tokenized_tweets = no_stopwords.apply(bigrams).apply(list)
    #Removes puntuation and stopwords
    
    corpus = sum(tokenized_tweets, [])

    
    bigram_counts = Counter(corpus)
    top_N_bigrams = bigram_counts.most_common(N)
    

    return(dict(top_N_bigrams))

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

def get_bigram_weight(tweet, dictionary, stop_words=stopwords.words('english')):
    """
    Calculates the weight of a tweet based on the presence of bigrams in a dictionary.

    Parameters:
    tweet (str): The tweet text.
    dictionary (dict): A dictionary containing bigrams as keys and their corresponding weights as values.
    stop_words (list, optional): A list of stop words to be excluded from the tweet. Defaults to the English stop words.

    Returns:
    float: The weight of the tweet based on the presence of bigrams in the dictionary.
    """

    tweet = tweet.lower().translate(str.maketrans('', '', (string.punctuation+"’0123456789"))).split()
    tweet = [word for word in tweet if word not in stop_words]
    tweet = list(bigrams(tweet))

    weight = sum([dictionary[word] if word in dictionary else 0 for word in tweet]) 

    return weight


class TweetCleaner(BaseEstimator,TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ##Filters only positive/negative tweets, drops na values and creates a binary label
        X = X.query("Sentiment == 'Positive' or Sentiment == 'Negative'")
        X = X.dropna(subset=["Content"])
        X["Sentiment_Label"] = X["Sentiment"].map({"Positive": 1, "Negative": 0})

        return X

class TweetFeatureExtractor(BaseEstimator,TransformerMixin):

    def __init__(self,N=100,m=100):
        self.N = N
        self.m = m

    def fit(self, X, y=None):
        #Creates the top N words and bigrams for positive and negative tweets
        self.positive_words = get_top_N_words(X,N=self.N,sentiment="Positive")
        self.negative_words = get_top_N_words(X,N=self.N,sentiment="Negative")
        self.positive_bigrams=get_top_N_bigrams(train_set,N=self.m,sentiment="Positive")
        self.negative_bigrams= get_top_N_bigrams(train_set,N=self.m,sentiment="Negative")



        return self

    def transform(self, X):

        #Creates the features for the dataset
        X["Word_Positive"] = X["Content"].apply(lambda x: get_word_weight(x,self.positive_words))
        X["Word_Negative"] = X["Content"].apply(lambda x: get_word_weight(x,self.negative_words))
        X["Bigram_Positive"] = X["Content"].apply(lambda x: get_bigram_weight(x,self.positive_bigrams))
        X["Bigram_Negative"] = X["Content"].apply(lambda x: get_bigram_weight(x,self.negative_bigrams))

        X["Length"]=X["Content"].apply(len)


        return X
    
class DF_Cleaner(BaseEstimator,TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
       
        Features = X.loc[:,["Word_Positive","Word_Negative","Bigram_Positive","Bigram_Negative","Length"]]
        Labels = X["Sentiment_Label"]
        
        return Features,Labels



if __name__ == "__main__":

    tweets_training = pd.read_csv("01_Data/Raw/twitter_training.csv",names=["ID","Entity","Sentiment","Content"])
    tweets_testing = pd.read_csv("01_Data/Raw/twitter_validation.csv",names=["ID","Entity","Sentiment","Content"])

    tweets = pd.concat([tweets_training,tweets_testing])
    train_set, test_set = train_test_split(tweets,test_size=0.2,random_state=42)


    N = int(sys.argv[1])
    ##We take 75% as many bigrams as words. This is quite the arbitrary decison
    m = int(N*0.75)

    tweet_pipeline = Pipeline([
        ('cleaner',TweetCleaner()),
        ('feature_extractor',TweetFeatureExtractor(N,m)),
        ("df_cleaner",DF_Cleaner())
    ])

    train_set,train_labels = tweet_pipeline.fit_transform(train_set)
    test_set,test_labels = tweet_pipeline.transform(test_set)
   



    train_set,validation_set,train_labels,validation_labels = train_test_split(train_set,train_labels,test_size=0.2,random_state=42)

    
    train_set.to_csv("01_Data/Processed/Training/training_set.csv", index=False)
    validation_set.to_csv("01_Data/Processed/Validation/validation_set.csv",index=False)
    test_set.to_csv("01_Data/Processed/Testing/testing_set.csv",index=False)

    train_labels.to_csv("01_Data/Processed/Training/training_labels.csv",index=False)
    validation_labels.to_csv("01_Data/Processed/Validation/validation_labels.csv",index=False)
    test_labels.to_csv("01_Data/Processed/Testing/testing_labels.csv",index=False)

    # validation_set.to_csv("../01_Data/Processed/validation_set.csv")
    # test_set.to_csv("../01_Data/Processed/testing_set.csv")

