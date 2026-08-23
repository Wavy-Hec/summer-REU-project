"""Bipartite railroad-county graph, drawn on two axes of a 3D plot.

Railroad companies sit along one axis and county codes along the other; an edge
means that company reported at least one accident in that county. Seeing both
node types laid out on their own axis is what makes the shared structure --
which counties tie which companies together -- visible at a glance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

from config import dataset_path

# Load the dataset
data_field = pd.read_csv(dataset_path(), low_memory=False)

# Remove missing values
data_field = data_field.dropna(subset=['Railroad Code', 'County Code'])

# Encode "Railroad Code" to numeric values, for use as an axis coordinate
data_field['Railroad Index'] = data_field['Railroad Code'].astype('category').cat.codes
data_field['County Index'] = pd.to_numeric(data_field['County Code'], errors='coerce')
data_field = data_field.dropna(subset=['County Index'])
data_field['County Index'] = data_field['County Index'].astype(int)

# Namespace the node ids before building the graph. Railroad indices and county
# codes are both plain integers, so an unprefixed bipartite edge list would
# silently merge a company and a county that happen to share a number.
data_field['Railroad Node'] = 'RR:' + data_field['Railroad Index'].astype(str)
data_field['County Node'] = 'CO:' + data_field['County Index'].astype(str)

# Create graph from pandas DataFrame
G = nx.from_pandas_edgelist(data_field, 'Railroad Node', 'County Node')

# Create a dictionary of positions: companies along x, counties along z
pos = {}
for node, index in zip(data_field['Railroad Node'], data_field['Railroad Index']):
    pos[node] = (index, 0)
for node, index in zip(data_field['County Node'], data_field['County Index']):
    pos[node] = (0, index)

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw nodes
for node in G.nodes():
    x, z = pos[node]
    is_railroad = node.startswith('RR:')
    ax.scatter(x, 0, z,
               color='tab:blue' if is_railroad else 'tab:orange',
               s=18 if is_railroad else 6,
               depthshade=False)

# Draw edges
for edge in G.edges():
    x = [pos[edge[0]][0], pos[edge[1]][0]]
    z = [pos[edge[0]][1], pos[edge[1]][1]]
    ax.plot([x[0], x[1]], [0, 0], [z[0], z[1]], color='grey', linewidth=0.2, alpha=0.3)

ax.set_xlabel('Railroad Code')
ax.set_zlabel('County Code')
ax.set_title('Railroad-county bipartite view')

print(f"Number of nodes: {G.number_of_nodes()} "
      f"({data_field['Railroad Node'].nunique()} railroads, "
      f"{data_field['County Node'].nunique()} counties)")
print(f"Number of edges: {G.number_of_edges()}")

plt.show()
