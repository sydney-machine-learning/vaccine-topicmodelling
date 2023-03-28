import pandas as df
from sentence_transformers import SentenceTransformer
import cuml
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bertopic import BERTopic
data_path = "../data/"
# file = ["preprocess_vaccination_all_tweets"
#         "preprocess_Tweets_Indonesia",
#         "preprocess_Tweets_Australia",
#         "preprocess_Tweets_Brazil",
#         "preprocess_Tweets_Japan",
#         "preprocess_Tweets_UK",
#         "preprocess_Tweets_US"]
file = ["Tweets_Indonesia"]

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
    

def main():
    for filename in file:
        print("\nProcessing: " + filename + '\n')
        ddf = df.read_csv(data_path + filename + '.csv',
                          dtype={'created_at':str, 'text':str},
                        usecols=["created_at", "text"],
                        sep = ',',
                        on_bad_lines='skip')
        ddf['text'] = ddf['text'].astype(str)
        ddf['text'].fillna("")
        preprocess_class = preprocess()

        ddf['text']=ddf.text.apply(lambda x :preprocess_class.preprocess_tweet(x))
        # print(ddf.text.head())
        print("Running sentence transformer")
        model = SentenceTransformer('distilbert-base-nli-mean-tokens',
                                    device='cuda')
        embeddings = model.encode(ddf['text'], show_progress_bar=True)

        print("Running UMAP")
        umap = cuml.manifold.UMAP(n_components=5, n_neighbors=5,
                                min_dist=0.0, random_state=12)
        reduced_data = umap.fit_transform(embeddings)

        print("Running HDBSCAN")
        clusterer = cuml.cluster.hdbscan.HDBSCAN(min_samples = 5, 
                                                min_cluster_size = 240,
                                                gen_min_span_tree=True)
        clusterer.fit(reduced_data)
        topic_model = BERTopic(umap_model=umap, hdbscan_model=clusterer)
        topics, probs = topic_model.fit_transform(ddf["text"])
        # soft_clusters = cuml.cluster.hdbscan.all_points_membership_vectors(clusterer)
        for i in range(20):
            print(topic_model.get_topic_info(i))
            print('\n')
        
        print(topic_model.get_topic(0))
        print(topic_model.get_topic(1))
        print(topic_model.get_topic(2))
        print(topic_model.get_topic(3))
        print(topic_model.get_topic(4))

if __name__ == "__main__":
    main()
