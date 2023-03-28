from nltk.corpus import stopwords

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

    
for filename in file:
    print("\nProcessing: " + filename + '\n')
    ddf = df.read_csv(data_path + filename + '.csv',
                      usecols=["text", "created_at"])

    ddf['text'] = ddf['text'].astype(str)
    ddf['text'].fillna("")
    preprocess_class = preprocess()

    ddf['text']=ddf.text.apply(lambda x :preprocess_class.preprocess_tweet(x))

    ddf.to_csv(data_path + 'preprocess_' + filename + '.csv')
