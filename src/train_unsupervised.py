# train_unsupervised.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from minisom import MiniSom

from data_preprocess import load_data, split_and_scale

os.makedirs('results/models', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

def run_kmeans(X_pos, k=3, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(X_pos)
    return km, labels

def run_som(X_pos, x=5, y=5, iters=200, random_seed=42):
    som = MiniSom(x, y, X_pos.shape[1], sigma=1.0, learning_rate=0.5, random_seed=random_seed)
    som.random_weights_init(X_pos)
    som.train_random(X_pos, iters)
    win_map = [som.winner(xi) for xi in X_pos]
    return som, win_map

if __name__ == '__main__':
    X, y = load_data()
    # Use scaled data if scaler exists, else perform split and scale here (no save)
    try:
        _, X_test = None, None
        # We'll just scale X_pos using StandardScaler via split_and_scale to be consistent
        X_train, X_test, y_train, y_test = split_and_scale(X, y)
        # combine back to get scaler-compatible positive examples
        # easier: get positives from scaled train+test arrays
        X_all_scaled = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])
        X_pos = X_all_scaled[y_all == 1]
    except Exception:
        # fallback: select positives from unscaled X (less ideal)
        X_pos = X[y == 1]

    print("Positive examples shape:", X_pos.shape)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X_pos)

    # KMeans
    km, labels = run_kmeans(Xp, k=3)
    plt.figure(figsize=(6,5))
    plt.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='viridis', s=10)
    plt.title('KMeans (k=3) on Positive examples (PCA proj)')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig('results/figures/kmeans_pos_pca.png')
    plt.close()
    joblib.dump(km, 'results/models/kmeans_pos.pkl')

    # SOM
    som, win_map = run_som(X_pos, x=5, y=5, iters=300)
    joblib.dump(som, 'results/models/som_pos.pkl')

    # Visualize SOM mapping by assigning integer label per winning cell
    idx_map = {}
    labels_som = []
    for w in win_map:
        if w not in idx_map:
            idx_map[w] = len(idx_map)
        labels_som.append(idx_map[w])
    labels_som = np.array(labels_som)

    # visualize SOM labels on PCA projection
    plt.figure(figsize=(6,5))
    plt.scatter(Xp[:,0], Xp[:,1], c=labels_som, cmap='tab10', s=10)
    plt.title('SOM clusters mapped to PCA projection')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig('results/figures/som_pos_pca.png')
    plt.close()

    print("Saved KMeans & SOM models and PCA visualizations in results/.")
