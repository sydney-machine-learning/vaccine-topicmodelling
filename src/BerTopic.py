from bertopic import BERTopic
import pandas as pd
# from sentence_transformers import SentenceTransformer


def main():
    df = pd.read_csv("Tweets_Indonesia.csv", dtype={'text': str})
    doc = df['text']

    # embeddings = model.encode(doc, show_progress_bar=True)
    topic_model = BERTopic(verbose=True, nr_topics="auto")
    topics, probs = topic_model.fit_transform(doc)
    topic_model.get_topic_info()


if __name__ == "__main__":
    main()
