import dask.dataframe as df
from sentence_transformers import SentenceTransformer
import cuml
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "../data/"
file = ["preprocess_Tweets_Indonesia",
        "preprocess_Tweets_Australia",
        "preprocess_Tweets_Brazil",
        "preprocess_Tweets_Japan",
        "preprocess_Tweets_UK",
        "preprocess_Tweets_US"]


filename = "preprocess_Tweets_Indonesia"


def main():
    print("reading from parquet")
    ddf = df.read_parquet(data_path + filename + '.parquet')

    # embeddings = model.encode(doc, show_progress_bar=True)
    print("Running sentenct transformer")
    model = SentenceTransformer('distilbert-base-nli-mean-tokens',
                                device='cuda')
    embeddings = model.encode(ddf['text'], show_progress_bar=True)

    print("Running UMAP")
    umap = cuml.manifold.UMAP(n_components=5, n_neighbors=15,
                              min_dist=0.0, random_state=12)
    reduced_data = umap.fit_transform(embeddings)

    print("Running HDBSCAN")
    clusterer = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=50,
                                             metric='euclidean',
                                             prediction_data=True)
    clusterer.fit(reduced_data)
    soft_clusters = cuml.cluster.hdbscan.all_points_membership_vectors(clusterer)
    print(ddf['text'].loc[clusterer.labels_ == 9].head())


if __name__ == "__main__":
    main()
