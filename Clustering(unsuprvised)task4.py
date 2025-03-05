import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Step 1: Generate Synthetic Data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.05, random_state=42)

# Step 2: Find Optimal K using Elbow Method
wcss = []  # Within-Cluster Sum of Squares
silhouette_scores = []  # Silhouette Scores
K_range = range(2, 10)  # K from 2 to 9

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # Store WCSS
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))  # Store Silhouette Score

# Step 3: Plot Elbow Method Graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')

# Step 4: Plot Silhouette Score Graph
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', linestyle='--', color='r')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal K')

plt.show()

# Step 5: Apply K-Means Clustering with Best K (Assuming K=3 from Elbow Method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)

# Step 6: Visualize Clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.title(f"K-Means Clustering (K={optimal_k})")
plt.legend()
plt.show()
print("Hierarchical Clustering & DBSCAN")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.05, random_state=42)

# Scatter plot of data points
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7)
plt.title("Generated Data for Clustering")
plt.show()
# Compute the linkage matrix
linked = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
# Apply Agglomerative Clustering with 3 clusters (from dendrogram)
agglo = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agglo.fit_predict(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title("Agglomerative Clustering Results")
plt.show()


from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate non-linearly separable data
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Scatter plot of data points
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7)
plt.title("Generated Data for DBSCAN Clustering")
plt.show()
# Apply DBSCAN with eps=0.2 and min_samples=5
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot DBSCAN Clustering Results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title("DBSCAN Clustering Results")
plt.show()


