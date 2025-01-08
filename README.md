代码使用方法：
1. 下载细胞模版
https://zenodo.org/records/6106718
2. 环境配置
```shell
$ conda create -n myenv python=3.9
$ conda activate myenv
(myenv) $ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
(myenv) $ conda install transformers
(myenv) $ conda install scikit-learn
(myenv) $ pip install sim-fish
(myenv) $ pip install umap-learn
(myenv) $ pip install seaborn
# PointMamba的环境配置详见 https://github.com/LMD0311/PointMamba/blob/main/USAGE.md
```

3. 图像模态
```shell
# 模拟数据
python image/sim_cell.py
# 训练
python image/train.py
# 可视化
python image/visualization.py
```

4. 点云模态
```shell
# 模拟数据
python pointcloud/sim_cell.py
# 训练
git clone https://github.com/LMD0311/PointMamba.git
# 替换掉tools/runner_finetune.py
python train.py
# 可视化
python pointcloud/pca_test1.py
python pointcloud/pca_test2.py
```