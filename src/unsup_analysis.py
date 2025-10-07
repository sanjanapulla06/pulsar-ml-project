# unsup_analysis.py
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from data_preprocess import load_data, split_and_scale

# load scaled positives (same approach as train_unsupervised)
X, y = load_data()
X_train, X_test, y_train, y_test = split_and_scale(X, y)
X_all = np.vstack([X_train, X_test])
y_all = np.hstack([y_train, y_test])
X_pos = X_all[y_all == 1]

# load KMeans if exists
try:
    km = joblib.load('results/models/kmeans_pos.pkl')
    labels_km = km.predict(X_pos if X_pos.shape[1]==km.cluster_centers_.shape[1] else PCA(n_components=km.cluster_centers_.shape[1]).fit_transform(X_pos))
    sil_km = silhouette_score(X_pos if X_pos.shape[1]==km.cluster_centers_.shape[1] else PCA(n_components=2).fit_transform(X_pos), labels_km)
    print("KMeans silhouette (approx):", sil_km)
    # sizes and centroids
    unique, counts = np.unique(labels_km, return_counts=True)
    print("KMeans cluster sizes:", dict(zip(unique, counts)))
    try:
        print("KMeans centroids (PCA-space or feature-space):")
        print(km.cluster_centers_)
    except Exception:
        pass
except FileNotFoundError:
    print("KMeans model not found at results/models/kmeans_pos.pkl")

# load SOM if exists (we can't easily compute silhouette on SOM directly but we can use mapped labels)
try:
    som = joblib.load('results/models/som_pos.pkl')
    from minisom import MiniSom
    # map each sample to a winning node index
    win_map = [som.winner(xi) for xi in X_pos]
    # convert wins to integer cluster ids
    idx_map = {}
    labels_som = []
    for w in win_map:
        if w not in idx_map:
            idx_map[w] = len(idx_map)
        labels_som.append(idx_map[w])
    labels_som = np.array(labels_som)
    sil_som = silhouette_score(X_pos, labels_som)
    print("SOM silhouette (approx):", sil_som)
    unique, counts = np.unique(labels_som, return_counts=True)
    print("SOM cluster sizes:", dict(zip(unique, counts)))
except FileNotFoundError:
    print("SOM model not found at results/models/som_pos.pkl")
except Exception as e:
    print("SOM analysis error:", e)
