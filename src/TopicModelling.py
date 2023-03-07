import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.decomposition import PCA


def umap(embeddings):
  umap_embeddings = umap.UMAP(n_neighbors=15,
                              n_components=5,
                              metric='cosine').fit_transform(embeddings)

def pca(embeddings, n_component):
  pca_n = PCA(n_components=n_component)
  pca_n.fit(embeddings)
  return pca_n.transform(embeddings)

def main():
  # read data
  print("Reading data")
  df = pd.read_csv('Tweets_UK.csv')
  tweets = df['text']

  print("Running BERT")
  model = SentenceTransformer('distilbert-base-nli-mean-tokens', device="cuda" )
  # pool = model.start_multi_process_pool()  
  embeddings = model.encode(tweets, show_progress_bar=True )
  print(embeddings)

  print("Running PCA")
  # pca_embeddings = pca(embeddings, 30)

  #cluster = hdbscan.HDBSCAN(min_cluster_size=15,
   #                       metric='euclidean',
    #                      cluster_selection_method='eom').fit(pca_embeddings)

if __name__ == "__main__":
  main()
