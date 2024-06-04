import os
import sys
import pickle
import cv2 as cv
import numpy as np
from glob import glob
from .utils import save_points, load_points, load_camera, load_manual_points,find_scene_file
from .calib import create_undistort_point_function, create_undistort_fisheye_point_function, \
 triangulate_points_fisheye, project_points_fisheye
from .plotting import plot_calib_board, plot_optimized_states, \
    plot_extrinsics, Cheetah
from .points import find_corners_images


def extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len, window_size=11, remove_unused_images=False):
    print(f"Finding calibration board corners for images in {img_dir}")
    filepaths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
    points, fpaths, cam_res = find_corners_images(filepaths, board_shape, window_size=window_size)
    saved_fnames = [os.path.basename(f) for f in fpaths]
    saved_points = points.tolist()
    if remove_unused_images:
        for f in filepaths:
            if os.path.basename(f) not in saved_fnames:
                print(f"Removing {f}")
                os.remove(f)
    save_points(out_fpath, saved_points, saved_fnames, board_shape, board_edge_len, cam_res)
# ==========  PLOTTING  ==========

def plot_corners(points_fpath):
    points, fnames, board_shape, board_edge_len, cam_res = load_points(points_fpath)
    plot_calib_board(points, board_shape, cam_res)


def plot_points_standard_undistort(points_fpath, camera_fpath):
    k, d, cam_res = load_camera(camera_fpath)
    points, _, board_shape, *_ = load_points(points_fpath)
    undistort_pts = create_undistort_point_function(k, d)
    undistorted_points = undistort_pts(points).reshape(points.shape)
    plot_calib_board(undistorted_points, board_shape, cam_res)


def plot_points_fisheye_undistort(points_fpath, camera_fpath):
    k, d, cam_res = load_camera(camera_fpath)
    points, _, board_shape, *_ = load_points(points_fpath)
    undistort_pts = create_undistort_fisheye_point_function(k, d)
    undistorted_points = undistort_pts(points).reshape(points.shape)
    plot_calib_board(undistorted_points, board_shape, cam_res)
    
    
def plot_scene(data_dir, scene_fname=None, manual_points_only=False, **kwargs):
    *_, scene_fpath = find_scene_file(data_dir, scene_fname, verbose=False)
    points_dir = os.path.join(os.path.dirname(scene_fpath), "points")
    pts_2d, frames = [], []
    if manual_points_only:
        points_fpaths = os.path.join(points_dir, "manual_points.json")
        pts_2d, frames, _ = load_manual_points(points_fpaths)
        pts_2d = pts_2d.swapaxes(0, 1)
        frames = [frames]*len(pts_2d)
    else:
        points_fpaths = sorted(glob(os.path.join(points_dir, 'points[1-9].json')))
        for fpath in points_fpaths:
            img_pts, img_names, *_ = load_points(fpath)
            pts_2d.append(img_pts)
            frames.append(img_names)

    plot_extrinsics(scene_fpath, pts_2d, frames, triangulate_points_fisheye, manual_points_only, **kwargs)
    
    
def plot_cheetah_states(states, smoothed_states=None, out_fpath=None, mplstyle_fpath=None):
    fig, axs = plot_optimized_states(states, smoothed_states, mplstyle_fpath)
    if out_fpath is not None:
        fig.savefig(out_fpath, transparent=True)
        print(f'Saved {out_fpath}\n')

        
def _plot_cheetah_reconstruction(positions, data_dir, scene_fname=None, labels=None, **kwargs):
    positions = np.array(positions)
    *_, scene_fpath = find_scene_file(data_dir, scene_fname, verbose=False)
    ca = Cheetah(positions, scene_fpath, labels, project_points_fisheye, **kwargs)
    ca.animation()
    
    
def plot_cheetah_reconstruction(data_fpath, scene_fname=None, **kwargs):
    label = os.path.basename(os.path.splitext(data_fpath)[0]).upper()
    with open(data_fpath, 'rb') as f:
        data = pickle.load(f)
    positions = data["smoothed_positions"] if 'EKF' in label else data["positions"]
    _plot_cheetah_reconstruction([positions], os.path.dirname(data_fpath), scene_fname, labels=[label], **kwargs)
    

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
