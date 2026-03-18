'''
Docstring for task.ipynb

Mall Customer Segmentation

Create dataset with:

Age
Annual Income
Spending Score

Steps

Create dataframe

Apply KMeans

Plot clusters

Display centroids

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = {
    "Age":            [19, 21, 20, 23, 31, 44, 45, 46, 47, 55, 60, 58, 35, 36, 38],
    "Annual Income":  [15, 16, 14, 18, 40, 85, 88, 90, 95, 60, 62, 65, 55, 50, 58],
    "Spending Score": [39, 81, 77, 40, 61, 20, 18, 15, 25, 49, 50, 48, 72, 75, 70]
}
df = pd.DataFrame(data)
print(df)
plt.scatter(df["Annual Income"], df["Spending Score"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Mall Customer Data")
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)
labels    = kmeans.labels_
centroids = kmeans.cluster_centers_
print("\nCluster Labels:", labels)
print("\nCentroids:\n", centroids)
plt.scatter(df["Annual Income"], df["Spending Score"], c=labels, cmap="viridis")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("KMeans Clustering — Mall Customers")
plt.show()
plt.scatter(df["Annual Income"], df["Spending Score"], c=labels, cmap="viridis")
plt.scatter(centroids[:, 1], centroids[:, 2], marker="X", s=200, c="red", label="Centroids")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("KMeans Clustering with Centroids")
plt.legend()
plt.show()