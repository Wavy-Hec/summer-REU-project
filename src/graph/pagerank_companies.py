"""Rank railroad companies by PageRank over an accident-to-company graph.

Builds a directed bipartite graph in which each accident (keyed by report year
and county) points at the company that reported it, runs PageRank over it, and
prints each company's accident count alongside its score. Also plots per-company
accident counts over time.

Runtime note: this iterates the full dataset row by row and then scans every
node once per company, so expect several minutes on the complete CSV.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config import dataset_path

# Load the dataset
data_field = pd.read_csv(dataset_path(), low_memory=False)

# Drop rows with no reporting company -- they cannot be ranked, and a missing
# code would become a float node that breaks the prefix match below.
data_field = data_field.dropna(subset=['Railroad Code'])

# Create an Accident ID combining 'Report Year' and 'County Code'
data_field['Accident_ID'] = data_field['Report Year'].astype(str) + "_" + data_field['County Code'].astype(str)

# Create a dictionary to hold the number of accidents per company
accidents_dict = {}

# Create an empty directed graph
G = nx.DiGraph()

# Add edges to the graph and count the number of accidents per company
for _, row in data_field.iterrows():
    G.add_edge(row['Accident_ID'], row['Railroad Code'])
    if row['Railroad Code'] in accidents_dict:
        accidents_dict[row['Railroad Code']] += 1
    else:
        accidents_dict[row['Railroad Code']] = 1

# Compute PageRank
pagerank = nx.pagerank(G)

# Compute the average PageRank for each company
# NOTE: this matches any node whose id starts with the company code, so company
# codes that are prefixes of one another share a score. See the README's
# "Known limitations" section.
avg_pagerank = {company: np.mean([pagerank[node] for node in G.nodes() if str(node).startswith(company)]) for company in accidents_dict.keys()}

for company in accidents_dict.keys():
    print(f"Company: {company}")
    print(f"Number of Accidents: {accidents_dict[company]}")
    print(f"Average PageRank: {avg_pagerank[company]}")
    print()

# Group data by 'Report Year' and 'Railroad Code' and count the number of accidents for each group
accidents_by_year = data_field.groupby(['Report Year', 'Railroad Code']).size().reset_index(name='Accident Count')

# Get unique company codes
company_codes = data_field['Railroad Code'].unique()

# Plot number of accidents over time for each company
plt.figure(figsize=(10, 6))
for company in company_codes:
    company_data = accidents_by_year[accidents_by_year['Railroad Code'] == company]
    plt.plot(company_data['Report Year'], company_data['Accident Count'], label=company)
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents Over Time')
plt.tight_layout()
plt.show()
