from nltk.corpus import stopwords
import re
import gc
import pandas as df

data_path = "../data/"

# file = ["vaccination_all_tweets",
#         "Tweets_Indonesia",
#         "Tweets_Australia",
#         "Tweets_Brazil",
#         "Tweets_Japan",
#         "Tweets_UK",
#         "Tweets_US"]
file = ["vaccination_all_tweets"]

class preprocess():
    def __init__(self):
        return

    def emoji_remove(self, tweet):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
              "]+", re.UNICODE)
        return emoji_pattern.sub(r'', tweet)

    def rt_remove(self, tweet):
        for word in tweet.split():
            if word[0] == '@':
                tweet = tweet.replace(word, "")
            if word == "RT":
                tweet = tweet.replace(word, "")
        return tweet

    def hashtag_remove(self, tweet):
        return re.sub(r'\#w+', '', tweet)

    def preprocess_tweet(self, tweets):
        tweets = self.rt_remove(tweets)
        tweets = self.emoji_remove(tweets)
        tweets = self.hashtag_remove(tweets)
        # print(tweets)
        return tweets
    

