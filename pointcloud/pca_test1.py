import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import argparse
from umap.umap_ import UMAP
from tools import builder
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.cm as cm
from utils import parser
from utils.config import *
import scanpy as sc

args = parser.get_args()
config = get_config(args)

# gene_mapping = {'ABL1': 0, 'AP2B1': 1, 'ATP1A1': 2, 'CCM2': 3, 'CLU': 4, 'DYNC1H1': 5, 'EEF2': 6, 'ERC1': 7, 'GALNT2': 8, 'HNRNPA2B1': 9, 'IQGAP1': 10, 'MCM2': 11, 'MTDH': 12, 'MYL9': 13, 'NCAPG': 14, 'NFIC': 15, 'PLEC': 16, 'PRPF31': 17, 'RPL36A': 18, 'RPL37': 19, 'RPS3': 20, 'RPS28': 21, 'SLC7A5': 22, 'SLC16A1': 23, 'SPTAN1': 24, 'SRRM2': 25, 'TNPO1': 26, 'TSHZ1': 27, 'TYMS': 28, 'ZBED4': 29}
# folder_mapping = {0: 'ABL1', 1: 'AP2B1', 2: 'ATP1A1', 3: 'CCM2', 4: 'CLU', 5: 'DYNC1H1', 6: 'EEF2', 7: 'ERC1', 8: 'GALNT2', 9: 'HNRNPA2B1', 10: 'IQGAP1', 11: 'MCM2', 12: 'MTDH', 13: 'MYL9', 14: 'NCAPG', 15: 'NFIC', 16: 'PLEC', 17: 'PRPF31', 18: 'RPL36A', 19: 'RPL37', 20: 'RPS3', 21: 'RPS28', 22: 'SLC7A5', 23: 'SLC16A1', 24: 'SPTAN1', 25: 'SRRM2', 26: 'TNPO1', 27: 'TSHZ1', 28: 'TYMS', 29: 'ZBED4'}
gene_mapping = {'cell_edge': 0, 'extranuclear': 1, 'foci': 2, 'intranuclear': 3, 'nuclear_edge': 4, 'pericellular': 5, 'perinuclear': 6, 'random': 7}
folder_mapping = {0: 'cell_edge', 1: 'extranuclear', 2: 'foci', 3: 'intranuclear', 4: 'nuclear_edge', 5: 'pericellular', 6: 'perinuclear', 7: 'random'}
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class CSVFileDataset(Dataset):
    def __init__(self, file_list, data_folder, train):
        """
        初始化 Dataset，传入文件列表和文件夹路径
        """
        self.file_list = file_list
        self.data_folder = data_folder
        self.train = train

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        根据索引 idx 读取对应的 CSV 文件并返回其内容作为一个样本
        """
        file_name = f"data_{idx}.csv"
        file_path = os.path.join(self.data_folder, file_name)

        # 读取CSV文件内容，这里假设每个文件就是特征数据
        data = pd.read_csv(file_path)
        if self.train:
            rotate_prob = random.random()
            if rotate_prob < 0.25:
                pass
            elif rotate_prob < 0.5:
                data['x'], data['y'] = -data['y'], data['x']
            elif rotate_prob < 0.75:
                data['x'], data['y'] = -data['x'], -data['y']
            else:
                data['x'], data['y'] = data['y'], -data['x']
        data['z'] = 0
        features = np.array(data[['x', 'y', 'z']].values)  # (N, 3) 形式的 numpy 数组

        features = pc_normalize(features)  # 归一化到单位球面
        # 提取标签，假设所有行的 label 是一样的，取第一行即可
        label = gene_mapping[data['label'].iloc[0]]

        # 将数据转换为torch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'coords': features_tensor,
            'label': label_tensor
        }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_dataset = CSVFileDataset(file_list=range(1024), data_folder='simulate_data/pointcloud/val', train=False)
test_dataset = CSVFileDataset(file_list=range(1024), data_folder='simulate_data/pointcloud/test', train=False)
model = builder.model_builder(config.model)
model.cls_head_finetune = nn.Linear(384, 8)
model.load_state_dict(torch.load('pattern_best.pth'))  # 加载模型参数
model.cls_head_finetune = nn.Identity()
model = model.to(device)

# 使用DataLoader
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
model.eval()
features = []
labels = []
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        data, target = batch['coords'].to(device), batch['label'].to(device)
        output = model(data)
        cls_features = output.cpu().numpy()  # 提取backbone的embedding
        features.append(cls_features)
        batch_labels = target.cpu().numpy()  # 提取标签
        labels.append(batch_labels)
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        data, target = batch['coords'].to(device), batch['label'].to(device)
        output = model(data)
        cls_features = output.cpu().numpy()  # 提取backbone的embedding
        features.append(cls_features)
        batch_labels = target.cpu().numpy()  # 提取标签
        labels.append(batch_labels)
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

pca = PCA(n_components=40, random_state=0)
features = pca.fit_transform(features)
labels = [folder_mapping[label] for label in labels]

sc.settings.set_figure_params(dpi_save=2000)
adata = sc.AnnData(features)
adata.obs['labels'] = labels  # 添加leiden标签
sc.pp.neighbors(adata, n_neighbors=32, use_rep='X', method='umap', random_state=0)  # 使用UMAP计算邻居
sc.tl.leiden(adata, flavor='igraph', n_iterations=2, resolution=0.55, random_state=0)
sc.tl.umap(adata, random_state=0)  # 使用UMAP降维
sc.pl.umap(adata, color=['leiden'], title="UMAP with Leiden Clustering", size=5, save="umap_leiden.png")
sc.pl.umap(adata, color=['labels'], title="UMAP with Gene Labels", size=5, legend_loc='on data', legend_fontsize=5, save="umap_labels.png")