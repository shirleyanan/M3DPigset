import cv2 as cv
import numpy as np

def triangulate_points(img_pts_1, img_pts_2, k1, d1, r1, t1, k2, d2, r2, t2):
    pts_1 = img_pts_1.reshape((-1,1,2))
    pts_2 = img_pts_2.reshape((-1, 1, 2))
    pts_1 = cv.undistortPoints(pts_1, k1, d1)
    pts_2 = cv.undistortPoints(pts_2, k2, d2)
    p1 = np.hstack((r1, t1))
    p2 = np.hstack((r2, t2))
    pts_4d = cv.triangulatePoints(p1, p2, pts_1, pts_2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


def project_points(obj_pts, k, d, r, t):
    pts =  cv.projectPoints(obj_pts, r, t, k, d)[0].reshape((-1, 2))
    return pts


# ========== FISHEYE CAMERA MODEL ==========
def triangulate_points_fisheye(img_pts_1, img_pts_2, k1, d1, r1, t1, k2, d2, r2, t2):
    pts_1 = img_pts_1.reshape((-1, 1, 2))
    pts_2 = img_pts_2.reshape((-1, 1, 2))
    pts_1 = cv.fisheye.undistortPoints(pts_1, k1, d1)
    pts_2 = cv.fisheye.undistortPoints(pts_2, k2, d2)
    p1 = np.hstack((r1, t1))
    p2 = np.hstack((r2, t2))
    pts_4d = cv.triangulatePoints(p1, p2, pts_1, pts_2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


def project_points_fisheye(obj_pts, k, d, r, t):
    obj_pts_reshaped = obj_pts.reshape((-1, 1, 3))
    r_vec = cv.Rodrigues(r)[0]  # 罗德里格斯变换，将旋转矩阵表示变换为旋转向量表示

    # fx = k[0,0]
    # fy = k[1,1]
    # cx = k[0,2]
    # cy = k[1,2]
    # u = fx * obj_pts_reshaped[0][0,0]/obj_pts_reshaped[0][0,2] + cx
    # v = fy * obj_pts_reshaped[0][0,1]/obj_pts_reshaped[0][0,2] + cy
    pts = cv.fisheye.projectPoints(obj_pts_reshaped, r_vec, t, k, d)[0].reshape((-1, 2))   # 该函数计算3D点到2D的（内外参已知）投影，返回2D点  和 雅可比矩阵
    return pts
