# https://www.youtube.com/watch?v=QLIkOtKS4os
import os.path as osp
import open3d as o3d
import torch_geometric.transforms as T

import torch
from torch_geometric.data import Data, Dataset, download_url

class ICRADataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['sample_000001.ply', 'sample_000002.ply']

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt']

    def download(self):
        pass

    def process(self):
        val = []
        for idx, raw_path in enumerate(self.raw_paths):
            # Read data from `raw_path`.
            pcd = o3d.io.read_point_cloud(raw_path)
            points = torch.tensor(pcd.points)
            # y = read_grasp_pose(raw_data) # TODO
            y = torch.tensor([0, 0, 0])

            data = Data(pos=points, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            val += data
        # torch.save(val, osp.join(self.processed_dir, f'total.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'

    # def __repr__(self) -> str:
    #     return (f'{self.__class__.__name__}({len(self)}, '
    #             f'categories={self.categories})')


dataset = ICRADataset(root="./data/ICRA")
print(dataset)

transform = T.KNNGraph(k=6)
dataset2 = ICRADataset(root="./data/ICRA", pre_transform=T.KNNGraph(k=6))
print(dataset2)