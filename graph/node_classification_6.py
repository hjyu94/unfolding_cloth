# Install required packages.
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.


import torch
from torch.nn import Linear
import torch.nn.functional as F


# class MLP(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         torch.manual_seed(12345)
#         self.lin1 = Linear(dataset.num_features, hidden_channels) # 각 노드당 1433 size featrue 를 갖고 있었는데 이를 16개로 낮추고
#         self.lin2 = Linear(hidden_channels, dataset.num_classes) # 16개의 node feature 를 이용해서 각 노드가 어떤 클래스에 속해있는지 class 분류
#
#     def forward(self, x):
#         x = self.lin1(x)
#         x = x.relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#         return x
#
# model = MLP(hidden_channels=16)
# print(model)

#
#
# criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
#
# def train():
#       model.train()
#       optimizer.zero_grad()  # Clear gradients.
#       out = model(data.x)  # Perform a single forward pass.
#       loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
#       loss.backward()  # Derive gradients.
#       optimizer.step()  # Update parameters based on gradients.
#       return loss
#
# def test():
#       model.eval()
#       out = model(data.x)
#       pred = out.argmax(dim=1)  # Use the class with highest probability.
#       test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
#       test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
#       return test_acc
#
# for epoch in range(1, 201):
#     loss = train()
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
#
# test_acc = test()


from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes) # 7

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(hidden_channels=16)
print(model)

model.eval()

out = model(data.x, data.edge_index) # untrained model 을 가지고 output 을 뽑아 본 것
visualize(out, color=data.y)

model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')