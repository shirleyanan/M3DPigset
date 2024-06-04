import os
import json
import pickle
import numpy as np
from src.plotting import Cheetah
from src.utils import find_scene_file
from src.calib import project_points_fisheye

def _plot_cheetah_reconstruction(positions, data_dir, scene_fname=None, labels=None, **kwargs):
    positions = np.array(positions)
    *_, scene_fpath = find_scene_file(data_dir, scene_fname, verbose=False)
    ca = Cheetah(positions, scene_fpath, labels, project_points_fisheye, **kwargs)
    ca.animation()

def plot_multiple_cheetah_reconstructions(data_fpaths, scene_fname=None, **kwargs):
    positions = []
    labels = []
    for data_fpath in data_fpaths:
        label = os.path.basename(os.path.splitext(data_fpath)[0]).upper()
        with open(data_fpath, 'rb') as f:
            data = pickle.load(f)
        positions.append(data["smoothed_positions"] if 'EKF' in label else data["positions"])
        labels.append(label)
    _plot_cheetah_reconstruction(positions, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(data_fpath)))), scene_fname, labels, **kwargs)

if __name__ == '__main__':

    print('Plotting results...')
    path = r'E:\Shirley_Code\python_code\SRE-SOWO\Predicted_2D3D\dataset\big\Walk6'
    data_fpaths = [
        os.path.join(path, '3D', 'sba.pickle'),
                   ]
    scene_fname = r'E:\Shirley_Code\python_code\SRE-SOWO\Predicted_2D3D\extrinsic_calib\4_View_scene.json'
    plot_multiple_cheetah_reconstructions(data_fpaths, scene_fname=scene_fname, reprojections=False, dark_mode=False)