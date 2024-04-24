# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import torch
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}') # 188 different graphs
print(f'Number of features: {dataset.num_features}') # 7 dimensional node feature
print(f'Number of classes: {dataset.num_classes}') # classify each graph one out of 2 classes

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}') # 17
print(f'Number of edges: {data.num_edges}') # 38
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # 2.24
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print(dataset.data) # y=[1] one graph label
