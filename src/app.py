import os
import sys
import pickle
import cv2 as cv
import numpy as np
from glob import glob
from .vid import proc_video, VideoProcessorCV
from .utils import create_board_object_pts, save_points, load_points, \
    save_camera, load_camera, load_manual_points, load_dlc_points_as_df, \
    find_scene_file, save_optimised_cheetah, save_3d_cheetah_as_2d
from .sba import _sba_board_points, _sba_points,_soft_sba_points
from .calib import calibrate_camera, calibrate_fisheye_camera, \
    calibrate_pair_extrinsics, calibrate_pair_extrinsics_fisheye, \
    create_undistort_point_function, create_undistort_fisheye_point_function, \
    triangulate_points, triangulate_points_fisheye, \
    project_points, project_points_fisheye, \
    _calibrate_pairwise_extrinsics
from .plotting import plot_calib_board, plot_optimized_states, \
    plot_extrinsics, Cheetah
from .points import find_corners_images

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


# ==========  VIDS  ==========

def get_vid_info(path_dir, vid_extension='mp4'):
    """Finds a video specified in/by the path variable and returns its info
    
    :param path: Either a directory containing a video or the path to a specific video file
    """
    from errno import ENOENT
    
    orig_path = path_dir
    if not os.path.isfile(path_dir):
        files = sorted(glob(os.path.join(path_dir, f"*.{vid_extension}"))) # assume path is a dir that holds video file(s)
        if files:
            path_dir = files[0]
        else:
            raise FileNotFoundError(ENOENT, os.strerror(ENOENT), orig_path) # assume videos didn't open due to incorrect path
        
    vid = VideoProcessorCV(in_name=path_dir)
    vid.close()
    return (vid.width(), vid.height()), vid.fps(), vid.frame_count(), vid.codec()


def create_labeled_videos(video_fpaths, videotype="mp4", codec="mp4v", outputframerate=None, out_dir=None, draw_skeleton=False, pcutoff=0.5, dotsize=6, colormap='jet', skeleton_color='white'):
    from functools import partial
    from multiprocessing import Pool

    print('Saving labeled videos...')
    
    bodyparts = get_markers()
    bodyparts2connect = get_skeleton() if draw_skeleton else None

    if not video_fpaths:
        print("No videos were found. Please check your paths\n")
        return

    if out_dir is None:
        out_dir = os.path.relpath(os.path.dirname(video_fpaths[0]), os.getcwd())

    func = partial(proc_video, out_dir, bodyparts, codec, bodyparts2connect, outputframerate, draw_skeleton, pcutoff, dotsize, colormap, skeleton_color)

    with Pool(min(os.cpu_count(), len(video_fpaths))) as pool:
        pool.map(func,video_fpaths)
        
    print('Done!\n')
