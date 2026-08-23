"""Accidents by weather condition, and a weather-keyed company co-occurrence graph.

Two outputs:

1. A bar chart of accident counts per reported weather condition.
2. A graph where nodes are railroad companies and an edge joins two companies
   that reported accidents under the same weather condition in the same year,
   partitioned with spectral clustering.

This is the "weather view" of the same companies that ``accidents_by_year.py``
builds a "county view" of -- the pair is what makes the analysis multi-view.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import numpy as np

from config import dataset_path

# Load the dataset
data_field = pd.read_csv(dataset_path(), low_memory=False)

# Count the occurrences of each unique value in the 'Weather Condition' column
weather_counts = data_field['Weather Condition'].value_counts()

# Plot the counts
plt.figure(figsize=(10, 6))
weather_counts.plot(kind='bar')
plt.xlabel('Weather Condition')
plt.ylabel('Count')
plt.title('Count of Accidents by Weather Condition')
plt.tight_layout()
plt.show()

# Group by "Railroad Code", "Incident Year", and "Weather Condition Code" and count the number of occurrences
grouped_data = data_field.groupby(['Railroad Code', 'Incident Year', 'Weather Condition Code']).size().reset_index(name='Accident Count')

# For the network graph
# Create an empty undirected graph
G = nx.Graph()

# Add nodes to the graph
G.add_nodes_from(grouped_data['Railroad Code'].unique())

# Group the data by 'Incident Year' and 'Weather Condition Code', and create a list of 'Railroad Code' for each group
grouped_by_year_weather = grouped_data.groupby(['Incident Year', 'Weather Condition Code'])['Railroad Code'].apply(list).reset_index()

# Add edges to the graph
for railroad_codes in tqdm(grouped_by_year_weather['Railroad Code']):
    if len(railroad_codes) > 1:
        G.add_edges_from([(railroad_codes[i], railroad_codes[j]) for i in range(len(railroad_codes)) for j in range(i + 1, len(railroad_codes))])

# Print the number of nodes and edges
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Remove unconnected nodes
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

# Print the number of nodes and edges after removing unconnected nodes
print(f"Number of nodes after removing isolates: {G.number_of_nodes()}")
print(f"Number of edges after removing isolates: {G.number_of_edges()}")

# Convert the graph into an adjacency matrix
# NOTE: networkx renamed to_scipy_sparse_matrix -> to_scipy_sparse_array in 2.7
# and removed the old name in 3.0. Densify as float: scikit-learn rejects sparse
# input carrying int64 indices, which is what networkx hands back on 64-bit builds.
A = nx.to_scipy_sparse_array(G).toarray().astype(float)

# Apply Spectral Clustering
sc = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='discretize')
sc.fit(A)
labels = sc.labels_

# Draw the graph using a spring layout
pos = nx.spring_layout(G, iterations=100)  # positions for all nodes

# Draw the graph with node color indicating cluster
plt.figure(figsize=(12, 8))
nx.draw(G, pos, node_color=labels, cmap=plt.cm.tab10, node_size=120, with_labels=True, font_size=6)
plt.title('Railroad companies sharing a weather-year, spectrally clustered (k=2)')
plt.show()

print(f"Cluster sizes: {np.bincount(labels)}")
