import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


df_name = "path/to/the/dataset.csv" 
df = pd.read_csv(df_name)

df = df.dropna(subset=['text']).copy()

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.split()

df['tokens'] = df['text'].apply(tokenize)

'''
  Uncomment below to use Word2Vec embeddings
# '''
# w2v_model = Word2Vec(
#     sentences=df['tokens'],
#     vector_size=100,
#     window=4,
#     min_count=2,
#     workers=4,
#     sg=0
# )

# def embed_song(tokens, model):
#     vectors = [model.wv[word] for word in tokens if word in model.wv]
#     if len(vectors) == 0:
#         return np.zeros(model.vector_size)
#     return np.mean(vectors, axis=0)

# X_w2v = np.vstack(df['tokens'].apply(lambda t: embed_song(t, w2v_model)))

'''
  Uncomment below to use TF-IDF features
'''
# lyrics_as_strings = df['tokens'].apply(lambda tokens: " ".join(tokens))
# tfidf = TfidfVectorizer(max_features=200, stop_words='english')
# X_tfidf = tfidf.fit_transform(lyrics_as_strings)

# scaler = StandardScaler(with_mean=False)
# X_tfidf_scaled = scaler.fit_transform(X_tfidf)
# X_tfidf_scaled = X_tfidf_scaled.toarray()

print("Mean-Shift Hyperparameter Tuning:")
bandwidth_grid = [0.5, 1]  
best_score_ms = -1
best_bw = None
best_labels_ms = None

for bw in bandwidth_grid:
    ms = MeanShift(bandwidth=bw)
    ms.fit(X_w2v)
    labels = ms.labels_
    score = silhouette_score(X_w2v, labels)
    print(f"Bandwidth: {bw}, Silhouette Score: {score:.4f}")
    if score > best_score_ms:
        best_score_ms = score
        best_bw = bw
        best_labels_ms = labels

print(f"Best bandwidth: {best_bw}, Best silhouette: {best_score_ms:.4f}")


print("\nGMM Hyperparameter Tuning:")
components_grid = [5, 10]
best_score_gmm = -1
best_n = None
best_labels_gmm = None

for n in components_grid:
    gmm = GaussianMixture(n_components=n, random_state=13)
    gmm.fit(X_tfidf_scaled)
    labels = gmm.predict(X_tfidf_scaled)
    score = silhouette_score(X_tfidf_scaled, labels)
    print(f"Components: {n}, Silhouette Score: {score:.4f}")
    if score > best_score_gmm:
        best_score_gmm = score
        best_n = n
        best_labels_gmm = labels

print(f"Best n_components: {best_n}, Best silhouette: {best_score_gmm:.4f}")
