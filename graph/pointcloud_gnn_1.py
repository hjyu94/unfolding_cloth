from torch_geometric.datasets import GeometricShapes
import torch_geometric.transforms as T

dataset = GeometricShapes(root='data/GeometricShapes')
print(dataset)

data = dataset[0]
print(data)

dataset.transform = T.SamplePoints(num=256)
dataset.transform = T.Compose([T.SamplePoints(num=256), T.KNNGraph(k=6)])

data = dataset[0]
print(data)
