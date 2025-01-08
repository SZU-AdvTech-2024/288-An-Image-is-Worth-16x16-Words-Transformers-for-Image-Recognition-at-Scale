import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor, ViTModel,  Trainer, TrainingArguments
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='scatter')
    args = parser.parse_args()

    if args.mode == 'scatter':
        data_folder = './data/simulate_cell_scatter'
    else:
        data_folder = './data/simulate_cell_kde'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
        # 将图像归一化为与 ViT 预处理器相同的均值和标准差
        lambda x: processor(images=x, do_rescale=False, return_tensors="pt")["pixel_values"].squeeze()
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        # 将图像归一化为与 ViT 预处理器相同的均值和标准差
        lambda x: processor(images=x, do_rescale=False, return_tensors="pt")["pixel_values"].squeeze()
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, 'train'), transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, 'val'), transform=test_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, 'test'), transform=test_transform)

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=8).to(device)

    # 使用DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    loss_fn = nn.CrossEntropyLoss()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if 'classifier' not in n and not any(nd in n for nd in no_decay)],
            'lr': 2e-5,
        },
        # 对于 bias 和 LayerNorm，不应用权重衰减
        {
            'params': [p for n, p in param_optimizer if 'classifier' not in n and any(nd in n for nd in no_decay)],
            'lr': 2e-5,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in param_optimizer if 'classifier' in n],
            'lr': 2e-5  # 分类头的学习率要高于其他参数
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    best_acc = 0
    patience = 5
    epochs_no_improve = 0
    for epoch in range(300):
        model.train()
        accum_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).logits
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()
        avg_loss = accum_loss / len(train_loader)
        print("Epoch: {}, Avg Loss: {}".format(epoch + 1, avg_loss))
        model.eval()
        mse_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data).logits
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        val_acc = 100 * correct / total
        print(f'Accuracy of the model on the {len(test_dataset)} val images: {val_acc}%')
        if best_acc < val_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            print('save best model')
            torch.save(model.state_dict(), "pattern_8_model.pth")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping')
            break

    model.load_state_dict(torch.load("pattern_8_model.pth"))
    model.eval()
    mse_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data).logits
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_acc = 100 * correct / total
    print(f'Accuracy of the model on the {len(test_dataset)} test images: {test_acc}%')

