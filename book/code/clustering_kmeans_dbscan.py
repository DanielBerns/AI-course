from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import numpy as np

# Synthetic data generation for standalone execution
X_tr, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=42)

# Fit K-Means and DBSCAN on synthetic data
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_tr)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_tr)

score = silhouette_score(X_tr, kmeans.labels_)
print(f"K-Means Silhouette Score: {score:.3f}")
print(f"DBSCAN Clusters Found:   {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}")
