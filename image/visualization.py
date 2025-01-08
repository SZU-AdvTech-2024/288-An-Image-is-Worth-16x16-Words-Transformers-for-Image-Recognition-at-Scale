import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor, ViTModel,  Trainer, TrainingArguments
import pandas as pd
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import argparse
from umap.umap_ import UMAP
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import scanpy as sc
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='scatter')
    args = parser.parse_args()

    if args.mode == 'scatter':
        data_folder = './data/simulate_cell_scatter'
    else:
        data_folder = './data/simulate_cell_kde'

    folder_path_val = "./simulate_cell_point/val"
    # 获取所有子文件夹的名称，并将其映射到数字
    subfolders = sorted([f for f in os.listdir(folder_path_val) if os.path.isdir(os.path.join(folder_path_val, f))])
    folder_mapping = {i: subfolders[i] for i in range(len(subfolders))}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        lambda x: processor(images=x, do_rescale=False, return_tensors="pt")["pixel_values"].squeeze()
    ])
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, 'val'), transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, 'test'), transform=transform)
    test_dataset = ConcatDataset([val_dataset, test_dataset])
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=8).to(device)
    model.load_state_dict(torch.load('pattern_8_model.pth'))  # 加载模型参数

    # 使用DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model.vit(data)
            last_hidden_state = output.last_hidden_state
            cls_features = last_hidden_state[:, 0, :].cpu().numpy()  # 提取backbone的embedding
            features.append(cls_features)
            batch_labels = target.cpu().numpy()  # 提取标签
            labels.append(batch_labels)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    pca = PCA(n_components=40, random_state=0)
    features = pca.fit_transform(features)
    labels_save = labels.copy()
    labels = [folder_mapping[label] for label in labels]

    sc.settings.set_figure_params(dpi_save=2000)
    adata = sc.AnnData(features)
    adata.obs['labels'] = labels  # 添加leiden标签
    sc.pp.neighbors(adata, n_neighbors=32, use_rep='X', method='umap', random_state=0)  # 使用UMAP计算邻居
    sc.tl.leiden(adata, flavor='igraph', n_iterations=2, resolution=0.1, random_state=0)
    sc.tl.umap(adata, random_state=0)  # 使用UMAP降维
    sc.pl.umap(adata, color=['labels'], title="UMAP Visualization", size=5, legend_loc='on data', legend_fontsize=5,save="umap_label.png")
    explained_variance_ratio = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 41), explained_variance_ratio, marker='o', linestyle='-', color='b')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, 41))  # 显示1到40的x坐标
    plt.show()
    plt.savefig(f"pca1to40_var.png")
    plt.close()
    features = features[:, :7]
    mean_features = []
    for label in range(8):
        idx = labels_save == label
        features_mean = features[idx].mean(axis=0)
        mean_features.append(features_mean)
    mean_features = np.array(mean_features)

    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_features, annot=False, cmap="coolwarm", xticklabels=[f"PC{i+1}" for i in range(7)], yticklabels=[folder_mapping[i] for i in range(8)])
    plt.title("Heatmap of PCA Features by Class")
    plt.xlabel("PCA Components")
    plt.ylabel("Labels")
    plt.tight_layout()
    plt.show()
    plt.savefig("pca10_heatmap.png")
