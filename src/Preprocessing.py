from nltk.corpus import stopwords
import pandas as pd
import gensim
import re 

# Assumption:
# 1. All "RT @are" removed, topics should purely based on the context
# 2. No need to process Captial or lower case since BERt handle this quite well
# 3. No need to lemmatize since this could reduce precision
# 4. emojis are removed since BERT is sequence relevent algorithm,
#   emojis are normally unsequenced, and the
# 5. No need to remove punctuations since BERT was trained with punctuations,
#   removing does not make differences


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


df = pd.read_csv('../data/Tweets_Indonesia.csv', dtype={'text': str})
df['text'] = df['text'].map(str)
df['text'].fillna("")
print(df.head())
preprocess_class = preprocess()
df['text'] = df['text'].apply(lambda x: preprocess_class.preprocess_tweet(x))

