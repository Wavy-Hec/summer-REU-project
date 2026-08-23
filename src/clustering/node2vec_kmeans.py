"""node2vec embeddings of the company graph, clustered with k-means.

Same company graph as ``spectral_kmeans.py`` -- companies joined when they share
a county, weighted by inverse difference in vehicle damage cost -- but instead of
a Laplacian eigendecomposition this learns 64-dimensional node2vec embeddings via
biased random walks, clusters them with k-means (k=4), and projects to 2D with
t-SNE for plotting.

Produces ``figures/kmeans_node2vec.svg``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from node2vec import Node2Vec
import seaborn as sns

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

# Use Node2Vec for graph embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

node_embeddings = np.array([model.wv[node] for node in G.nodes])

# Cluster the embeddings
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(node_embeddings)

# Create cluster assignments for the nodes
clusters = {node: kmeans.labels_[i] for i, node in enumerate(G.nodes)}

# Prepare for t-SNE transformation
node_ids = list(G.nodes)
n_samples = len(node_ids)
perplexity_value = min(30, n_samples - 1)

# Reduce dimensionality with t-SNE
# NOTE: scikit-learn renamed TSNE's n_iter -> max_iter in 1.5 and removed the
# old name in 1.7.
trans = TSNE(n_components=2, early_exaggeration=10,
             perplexity=perplexity_value, max_iter=1000, n_iter_without_progress=500,
             learning_rate=600.0, random_state=42)
node_embeddings_2d = trans.fit_transform(node_embeddings)

# Create dataframe for plotting
data_tsne = pd.DataFrame(zip(node_ids, list(node_embeddings_2d[:, 0]), list(node_embeddings_2d[:, 1]), [clusters[node] for node in node_ids]),
                         columns=['node_ids', 'x', 'y', 'kmeans'])

# Plot using seaborn.
plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne, x='x', y='y', hue='kmeans', palette="bright",
                alpha=0.55, s=200).set_title('Node2vec clusters with k-means')
FIGURES.mkdir(exist_ok=True)
plt.savefig(FIGURES / 'kmeans_node2vec.svg')
plt.show()

# Print out nodes and their associated clusters
for node, cluster in clusters.items():
    print(f"Node: {node}, Cluster: {cluster}")
