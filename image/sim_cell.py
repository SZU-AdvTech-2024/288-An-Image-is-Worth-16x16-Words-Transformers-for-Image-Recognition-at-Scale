import simfish
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='scatter')
    args = parser.parse_args()

    if args.mode == 'scatter':
        data_folder = './data/simulate_cell_scatter'
    else:
        data_folder = './data/simulate_cell_kde'

    path_template_directory = './templates'
    proportion_list = [0.6, 0.7, 0.8, 0.9]
    train_folder = os.path.join(data_folder, 'train')
    val_folder = os.path.join(data_folder, 'val')
    test_folder = os.path.join(data_folder, 'test')

    for pattern in ['random', 'foci', 'intranuclear', 'extranuclear', 'nuclear_edge', 'perinuclear', 'cell_edge', 'pericellular']:
        pattern_train_folder = os.path.join(train_folder, pattern)
        pattern_val_folder = os.path.join(val_folder, pattern)
        pattern_test_folder = os.path.join(test_folder, pattern)
        if not os.path.exists(pattern_train_folder):
            os.makedirs(pattern_train_folder)
        if not os.path.exists(pattern_val_folder):
            os.makedirs(pattern_val_folder)
        if not os.path.exists(pattern_test_folder):
            os.makedirs(pattern_test_folder)
        for train_idx in range(600):
            cell_idx = np.random.randint(1, int(0.6*318))
            proportion_idx = np.random.randint(len(proportion_list))
            sim_cell = simfish.simulate_localization_pattern(path_template_directory, 1024, i_cell=cell_idx, pattern=pattern, proportion_pattern=proportion_list[proportion_idx])
            cell_coord = sim_cell['cell_coord']
            nuc_coord = sim_cell['nuc_coord']
            points = list(zip(cell_coord[:, 1], cell_coord[:, 0]))
            cell_boundary_polygon = Polygon(points)
            points = list(zip(nuc_coord[:, 1], nuc_coord[:, 0]))
            nuclei_boundary_polygon = Polygon(points)
            boundaries = gpd.GeoDataFrame(geometry=[nuclei_boundary_polygon, cell_boundary_polygon])
            rna_coord = sim_cell['rna_coord'][:, 1:]
            boundaries.plot(edgecolor='black', facecolor='none', linewidth=1.5)
            if mode == 'scatter':
                plt.scatter(rna_coord[:, 1], rna_coord[:, 0], s=1, c='red', alpha=0.5)
            else:
                sns.kdeplot(x=rna_coord[:, 1], y=rna_coord[:, 0], cmap='viridis', fill=True, levels=20, alpha=0.3)
            plt.axis('off')
            plt.show()
            plt.savefig(os.path.join(pattern_train_folder, f'{pattern}_{train_idx}.png'), dpi=500, bbox_inches='tight')
            plt.close()
        for val_idx in range(200):
            cell_idx = np.random.randint(int(0.6*318), int(0.8*318))
            proportion_idx = np.random.randint(len(proportion_list))
            sim_cell = simfish.simulate_localization_pattern(path_template_directory, 1024, i_cell=cell_idx, pattern=pattern, proportion_pattern=proportion_list[proportion_idx])
            cell_coord = sim_cell['cell_coord']
            nuc_coord = sim_cell['nuc_coord']
            points = list(zip(cell_coord[:, 1], cell_coord[:, 0]))
            cell_boundary_polygon = Polygon(points)
            points = list(zip(nuc_coord[:, 1], nuc_coord[:, 0]))
            nuclei_boundary_polygon = Polygon(points)
            boundaries = gpd.GeoDataFrame(geometry=[nuclei_boundary_polygon, cell_boundary_polygon])
            rna_coord = sim_cell['rna_coord'][:, 1:]
            boundaries.plot(edgecolor='black', facecolor='none', linewidth=1.5)
            if mode == 'scatter':
                plt.scatter(rna_coord[:, 1], rna_coord[:, 0], s=1, c='red', alpha=0.5)
            else:
                sns.kdeplot(x=rna_coord[:, 1], y=rna_coord[:, 0], cmap='viridis', fill=True, levels=20, alpha=0.3)
            plt.axis('off')
            plt.show()
            plt.savefig(os.path.join(pattern_val_folder, f'{pattern}_{val_idx}.png'), dpi=500, bbox_inches='tight')
            plt.close()
        for test_idx in range(200):
            cell_idx = np.random.randint(int(0.8*318), 318)
            proportion_idx = np.random.randint(len(proportion_list))
            sim_cell = simfish.simulate_localization_pattern(path_template_directory, 1024, i_cell=cell_idx, pattern=pattern, proportion_pattern=proportion_list[proportion_idx])
            cell_coord = sim_cell['cell_coord']
            nuc_coord = sim_cell['nuc_coord']
            points = list(zip(cell_coord[:, 1], cell_coord[:, 0]))
            cell_boundary_polygon = Polygon(points)
            points = list(zip(nuc_coord[:, 1], nuc_coord[:, 0]))
            nuclei_boundary_polygon = Polygon(points)
            boundaries = gpd.GeoDataFrame(geometry=[nuclei_boundary_polygon, cell_boundary_polygon])
            rna_coord = sim_cell['rna_coord'][:, 1:]
            boundaries.plot(edgecolor='black', facecolor='none', linewidth=1.5)
            if mode == 'scatter':
                plt.scatter(rna_coord[:, 1], rna_coord[:, 0], s=1, c='red', alpha=0.5)
            else:
                sns.kdeplot(x=rna_coord[:, 1], y=rna_coord[:, 0], cmap='viridis', fill=True, levels=20, alpha=0.3)
            plt.axis('off')
            plt.show()
            plt.savefig(os.path.join(pattern_test_folder, f'{pattern}_{test_idx}.png'), dpi=500, bbox_inches='tight')
            plt.close()
