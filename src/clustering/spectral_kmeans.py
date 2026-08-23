"""Spectral clustering of railroad companies by shared vehicle damage cost.

Pipeline: build a company graph whose edges join companies operating in the same
county and whose weights are the inverse difference in their booked vehicle
damage cost, take the normalized Laplacian, keep its two smallest eigenvectors,
sweep k with k-means and pick the k with the best silhouette score, then project
to 2D with PCA for plotting.

Produces ``figures/kmeans_spectral_clustering.svg``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

from config import dataset_path

FIGURES = Path(__file__).resolve().parents[2] / "figures"

# Load the dataset
data_field = pd.read_csv(dataset_path(), low_memory=False)

# Get unique company codes and take the first 31
companies = data_field['Railroad Code'].unique()[:31]

# Filter the data to include only your 31 companies
data_field = data_field[data_field['Railroad Code'].isin(companies)]

# Group by 'County Code', 'Railroad Code' and calculate the sum of 'Vehicle Damage Cost'
grouped_data = data_field.groupby(['County Code', 'Railroad Code'])['Vehicle Damage Cost'].sum().reset_index()

# Encode the county names to numerical values
le = LabelEncoder()
grouped_data['Encoded County'] = le.fit_transform(grouped_data['County Code'])

# Initialize a directed graph
G = nx.DiGraph()

# Add nodes to the graph
for _, row in grouped_data.iterrows():
    G.add_node(row['Railroad Code'], cost=row['Vehicle Damage Cost'], county=row['County Code'])

# Add edges to the graph
for i in range(len(grouped_data)):
    for j in range(i + 1, len(grouped_data)):
        company1 = grouped_data.loc[i, 'Railroad Code']
        company2 = grouped_data.loc[j, 'Railroad Code']
        county1 = grouped_data.loc[i, 'County Code']
        county2 = grouped_data.loc[j, 'County Code']
        if county1 == county2:  # Only add an edge if the companies are in the same county
            diff = abs(G.nodes[company1]['cost'] - G.nodes[company2]['cost'])
            similarity_score = 1 / diff if diff != 0 else 0
            G.add_edge(company1, company2, weight=similarity_score)

# Convert the graph to an adjacency matrix.
# Symmetrize first: edges are added in one direction only, and both
# csgraph.laplacian(normed=True) and eigsh assume a symmetric matrix.
adj_matrix = nx.adjacency_matrix(G.to_undirected(), weight='weight').toarray()

# Compute the normalized Laplacian
laplacian = csgraph.laplacian(adj_matrix, normed=True)

# Compute eigenvalues and eigenvectors
num_eigenvectors = 2
eigenvalues, eigenvectors = eigsh(laplacian, k=num_eigenvectors, which='SM')

# Normalize the eigenvectors
eigenvectors = normalize(eigenvectors)

best_score = -1
best_k = 2

# Trying different numbers of clusters and find the best one based on silhouette score
max_k = min(11, len(eigenvectors))
for k in range(2, max_k):
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(eigenvectors[:, :2])

    score = silhouette_score(eigenvectors, labels)

    if score > best_score:
        best_score = score
        best_k = k

    print(f"For k={k}, silhouette score: {score}")

print(f"\nBest number of clusters (k): {best_k}")
print(f"Best silhouette score: {best_score}")

# Use best_k as the number of clusters for KMeans
kmeans = KMeans(n_clusters=best_k, n_init=10)
labels = kmeans.fit_predict(eigenvectors[:, :2])

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(eigenvectors)

node_labels = dict(zip(G.nodes, labels))

# Create a scatter plot to visualize the results
plt.figure(figsize=(8, 8))
for i in range(len(principalComponents)):
    plt.scatter(principalComponents[i][0], principalComponents[i][1], color=plt.cm.nipy_spectral(labels[i] / 10.), s=100)

plt.title(f'Spectral embedding, k-means (k={best_k})')
FIGURES.mkdir(exist_ok=True)
plt.savefig(FIGURES / 'kmeans_spectral_clustering.svg')
plt.show()

# Print out nodes and their associated clusters
for node, cluster in node_labels.items():
    print(f"Node: {node}, Cluster: {cluster}")
