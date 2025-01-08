import simfish
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import random
import os

path_template_directory = '../ViT/templates'
pc_data_folder = './simulate_data/pointcloud'
proportion = [0.6, 0.7, 0.8, 0.9]
pts_num = 1024

pc_train_idx = 0
pc_val_idx = 0
pc_test_idx = 0
pc_data_train_folder = os.path.join(pc_data_folder, f'train')
pc_data_val_folder = os.path.join(pc_data_folder, f'val')
pc_data_test_folder = os.path.join(pc_data_folder, f'test')
os.makedirs(pc_data_train_folder, exist_ok=True)
os.makedirs(pc_data_val_folder, exist_ok=True)
os.makedirs(pc_data_test_folder, exist_ok=True)
for pattern in ['random', 'foci', 'intranuclear', 'extranuclear', 'nuclear_edge', 'perinuclear', 'cell_edge', 'pericellular']:
    for cell_idx in range(0, int(0.6 * 318)):
        for time in range(2):
            prob = random.random()
            if prob < 0.25:
                proportion_select = 0
            elif prob < 0.5:
                proportion_select = 1
            elif prob < 0.75:
                proportion_select = 2
            else:
                proportion_select = 3
            sim_cell = simfish.simulate_localization_pattern(path_template_directory, pts_num, i_cell=cell_idx, pattern=pattern, proportion_pattern=proportion[proportion_select])
            rna_coord = sim_cell['rna_coord'][:, 1:]
            cell_coord = sim_cell['cell_coord']
            nuc_coord = sim_cell['nuc_coord']
            cell_coord = pd.DataFrame(cell_coord, columns=['y', 'x'])
            cell_coord = cell_coord.sample(n=600, random_state=42)
            nuc_coord = pd.DataFrame(nuc_coord, columns=['y', 'x'])
            nuc_coord = nuc_coord.sample(n=200, random_state=42)
            rna_coord = pd.DataFrame(rna_coord, columns=['y', 'x'])
            data = pd.concat([cell_coord, nuc_coord, rna_coord], axis=0)
            data['label'] = pattern
            data.to_csv(os.path.join(pc_data_train_folder, f'data_{pc_train_idx}.csv'), index=False)
            pc_train_idx += 1

    for cell_idx in range(int(0.6 * 318), int(0.8 * 318)):
        for time in range(2):
            prob = random.random()
            if prob < 0.25:
                proportion_select = 0
            elif prob < 0.5:
                proportion_select = 1
            elif prob < 0.75:
                proportion_select = 2
            else:
                proportion_select = 3
            sim_cell = simfish.simulate_localization_pattern(path_template_directory, pts_num, i_cell=cell_idx,
                                                             pattern=pattern, proportion_pattern=proportion[proportion_select])
            rna_coord = sim_cell['rna_coord'][:, 1:]
            cell_coord = sim_cell['cell_coord']
            nuc_coord = sim_cell['nuc_coord']
            cell_coord = pd.DataFrame(cell_coord, columns=['y', 'x'])
            cell_coord = cell_coord.sample(n=600, random_state=42)
            nuc_coord = pd.DataFrame(nuc_coord, columns=['y', 'x'])
            nuc_coord = nuc_coord.sample(n=200, random_state=42)
            rna_coord = pd.DataFrame(rna_coord, columns=['y', 'x'])
            data = pd.concat([cell_coord, nuc_coord, rna_coord], axis=0)
            data['label'] = pattern
            data.to_csv(os.path.join(pc_data_val_folder, f'data_{pc_val_idx}.csv'), index=False)
            pc_val_idx += 1
    for cell_idx in range(int(0.8 * 318), 318):
        for time in range(2):
            prob = random.random()
            if prob < 0.25:
                proportion_select = 0
            elif prob < 0.5:
                proportion_select = 1
            elif prob < 0.75:
                proportion_select = 2
            else:
                proportion_select = 3
            sim_cell = simfish.simulate_localization_pattern(path_template_directory, pts_num, i_cell=cell_idx,
                                                             pattern=pattern, proportion_pattern=proportion[proportion_select])
            rna_coord = sim_cell['rna_coord'][:, 1:]
            cell_coord = sim_cell['cell_coord']
            nuc_coord = sim_cell['nuc_coord']
            cell_coord = pd.DataFrame(cell_coord, columns=['y', 'x'])
            cell_coord = cell_coord.sample(n=600, random_state=42)
            nuc_coord = pd.DataFrame(nuc_coord, columns=['y', 'x'])
            nuc_coord = nuc_coord.sample(n=200, random_state=42)
            rna_coord = pd.DataFrame(rna_coord, columns=['y', 'x'])
            data = pd.concat([cell_coord, nuc_coord, rna_coord], axis=0)
            data['label'] = pattern
            data.to_csv(os.path.join(pc_data_test_folder, f'data_{pc_test_idx}.csv'), index=False)
            pc_test_idx += 1