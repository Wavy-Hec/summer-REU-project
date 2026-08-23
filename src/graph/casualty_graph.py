"""Company graph keyed on total casualties per county.

Sums killed and injured across both FRA reporting forms (57 and 55A) into a
single casualty count per company per county, then joins any two companies that
share a county. Node labels carry the casualty total, so the drawing doubles as
a severity readout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

from config import dataset_path

# Load the dataset
data_field = pd.read_csv(dataset_path(), low_memory=False)

# Combine the "Killed" and "Injured" columns to get total casualties
data_field['Total Casualties'] = data_field['Total Killed Form 57'] + data_field['Total Killed Form 55A'] + data_field['Total Injured Form 57'] + data_field['Total Injured Form 55A']

# Group by County Code and railroad code
grouped_data_county = data_field.groupby(['County Code', 'Railroad Code'])['Total Casualties'].sum().reset_index()

print(grouped_data_county)  # Print the grouped data

# Look the casualty total up once per company instead of re-scanning the frame
# inside the pair loop below.
casualties_by_company = (
    grouped_data_county.groupby('Railroad Code')['Total Casualties'].first().to_dict()
)


def label(company):
    return f"{company}({casualties_by_company[company]})"


# Initialize a graph
G_county = nx.Graph()

# Add nodes to the graph
for _, row in grouped_data_county.iterrows():
    G_county.add_node(f"{row['Railroad Code']}({row['Total Casualties']})", casualties=row['Total Casualties'])

# Add edges to the graph for companies in the same county
for county in grouped_data_county['County Code'].unique():
    same_county_companies = grouped_data_county[grouped_data_county['County Code'] == county]['Railroad Code']
    for company1, company2 in combinations(same_county_companies, 2):
        G_county.add_edge(label(company1), label(company2))

print(f"Number of nodes: {G_county.number_of_nodes()}")
print(f"Number of edges: {G_county.number_of_edges()}")

# Visualize the graph
plt.figure(figsize=(12, 8))
nx.draw(G_county, with_labels=True, node_size=300, font_size=6)
plt.title('Companies sharing a county, labelled with total casualties')
plt.show()
