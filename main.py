from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import debug
import correspondences
from logger import logger, log_time
import data_structure
import geometry
import open3d as o3d
import bundle_adjustment

from scipy.spatial.transform import Rotation

import sys
sys.path.append("build/Release")
import cpp_ba

K = np.array([[1660.076384971925, 0, 986.3176281647676], 
              [0, 1656.285062933074, 763.9663384634628], 
              [0, 0, 1]])


folder_name = "my"
recon = data_structure.Reconstruction()

logger.info("loading images")
im1 = np.asarray(Image.open(f"images/{folder_name}/frame01.png"))
im2 = np.asarray(Image.open(f"images/{folder_name}/frame02.png"))


putative1, putative2 = correspondences.find_correspondences_akaze(im1, im2)

with log_time("five-point algorithm"):
    # 5-point algorithm + RANSAC
    E, mask_E = cv2.findEssentialMat(putative1, putative2, K,  method=cv2.RANSAC, prob=0.9999, threshold=0.5)
    inlier_mask = mask_E.ravel().astype(bool)
    inliers1 = putative1[inlier_mask]
    inliers2 = putative2[inlier_mask]
    logger.info(f"inliers1.shape = {inliers1.shape}")
    logger.info(f"inliers2.shape = {inliers2.shape}")

with log_time("recover pose"):
    # Recover relative pose
    _, R, t, mask_pose = cv2.recoverPose(E, inliers1, inliers2, K)
    inlier_mask = mask_pose.ravel().astype(bool)
    inliers1 = inliers1[inlier_mask]
    inliers2 = inliers2[inlier_mask]
    logger.info(f"inliers1.shape = {inliers1.shape}")
    logger.info(f"inliers2.shape = {inliers2.shape}")

with log_time("normalize points"):
    # normalize points so that triangulate optimal works
    inliers1 = cv2.undistortPoints(inliers1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    inliers2 = cv2.undistortPoints(inliers2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

with log_time("add initial views, observations and points"):
    C1 = np.concat((np.identity(3), np.array([[0], [0], [0]])), axis = 1)
    C2 = np.concat((R, t), axis = 1)
    v1_id = recon.add_view(np.identity(3), np.array([0, 0, 0]), im1)
    v2_id = recon.add_view(R, t, im2)
    for p1, p2 in zip(inliers1, inliers2):
        point = geometry.triangulate_optimal(C1, C2, p1, p2)
        point_id = recon.add_point(point)
        recon.add_observation(p1, v1_id, point_id)
        recon.add_observation(p2, v2_id, point_id)

with log_time("packet info for BA"):
    camera_flat, point_flat, observations = bundle_adjustment.packet_data(recon)

with log_time("bundle Adjustment"):
    results = bundle_adjustment.bundle_adjustment(camera_flat, point_flat, observations)
    camera_flat = results.cameras
    point_flat = results.points

with log_time("unpack BA"):
    bundle_adjustment.unpack_results(recon, camera_flat, point_flat, observations)


points = []
for p in recon.points:
    points.append(recon.points[p].xyz)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
