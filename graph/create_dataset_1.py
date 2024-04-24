import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

import open3d as o3d
import torch_geometric.transforms as T
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


# 1. 포인트 클라우드 파일 불러오기
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = torch.tensor(pcd.points)
    # colors = np.asarray(pcd.colors)
    return points

# 3. Torch Geometry 데이터셋 생성
def create_dataset(points_tensor):
    # 데이터를 torch_geometric.data.Data 형식으로 변환
    data = Data(pos=points_tensor)  # pos는 포인트 클라우드의 좌표를 나타냄
    data = T.KNNGraph(k=6, force_undirected=True)(data)
    return data

# 4. 학습 및 테스트
# 데이터셋 로더 생성
dataset = load_point_cloud("/graph/data/ICRA/raw/extracted_pcd.ply")
dataset = create_dataset(dataset)
visualize(dataset, 1)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 이제 이 loader를 사용하여 GNN 모델을 학습하고 평가할 수 있습니다.
# 예를 들어 PyTorch Geometric의 GNN 모델을 사용하여 학습 및 평가를 진행할 수 있습니다.