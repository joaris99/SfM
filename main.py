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

with log_time("load images"):
    images = []
    for i in range(36):
        if i < 9:
            im = np.asarray(Image.open(f"images/{folder_name}/frame0{i + 1}.png"))
        else:
            im = np.asarray(Image.open(f"images/{folder_name}/frame{i + 1}.png"))
        images.append(im)
    im1 = images[0]
    im2 = images[1]

with log_time("find correspondences"):
    kp1, kp2, desc1, desc2, matches = correspondences.find_correspondences_akaze(im1, im2)
    pts1, pts2, idx1, idx2 = correspondences.get_coordinates(kp1, kp2, matches)
  



with log_time("five-point algorithm"):
    # 5-point algorithm + RANSAC
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K,  method=cv2.RANSAC, prob=0.9999, threshold=0.5)
    mask_E = mask_E.ravel().astype(bool)
    matches = matches[mask_E]
    pts1 = pts1[mask_E]
    pts2 = pts2[mask_E]
    idx1 = idx1[mask_E]
    idx2 = idx2[mask_E]
    # logger.info(f"inliers1.shape = {inliers1.shape}")
    # logger.info(f"inliers2.shape = {inliers2.shape}")

with log_time("recover pose"):
    # Recover relative pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    mask_pose = mask_pose.ravel().astype(bool)
    matches = matches[mask_pose]
    pts1 = pts1[mask_pose]
    pts2 = pts2[mask_pose]
    idx1 = idx1[mask_pose]
    idx2 = idx2[mask_pose]
    # logger.info(f"inliers1.shape = {inliers1.shape}")
    # logger.info(f"inliers2.shape = {inliers2.shape}")

with log_time("normalize points"):
    # normalize points so that triangulate optimal works
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

with log_time("add initial views, observations and points"):
    C1 = np.concat((np.identity(3), np.array([[0], [0], [0]])), axis = 1)
    C2 = np.concat((R, t), axis = 1)
    v1_id = recon.add_view(np.identity(3), np.array([0, 0, 0]), kp1, desc1, im1)
    v2_id = recon.add_view(R, t, kp2, desc2, im2)
    

    for i in range(len(idx1)):

        # p1 = np.array(kp1[idx1[i]].pt)
        # p2 = np.array(kp2[idx2[i]].pt)

        t1 = pts1_norm[i]
        t2 = pts2_norm[i]

        point = geometry.triangulate_optimal(C1, C2, t1, t2)
        point_id = recon.add_point(point)

        recon.add_observation(
            xy=t1,
            view_id=v1_id,
            point_id=point_id,
            feature_idx=idx1[i]
        )

        recon.add_observation(
            xy=t2,
            view_id=v2_id,
            point_id=point_id,
            feature_idx=idx2[i]
        )


# for iteration in range(1):
for iteration in range(34):
    
    with log_time("packet info for BA"):
        camera_flat, point_flat, observations = bundle_adjustment.packet_data(recon)

    with log_time("bundle Adjustment"):
        results = bundle_adjustment.bundle_adjustment(camera_flat, point_flat, observations)
        camera_flat = results.cameras
        point_flat = results.points

    with log_time("unpack BA"):
        bundle_adjustment.unpack_results(recon, camera_flat, point_flat, observations)
    
    debug.plot_3D(recon)
    
    im_prev = images[iteration + 1]
    im_next = images[iteration + 2]

    
    prev_view = recon.views[iteration + 1]

    with log_time("find correspondences"):
        kp_prev, kp_next, desc_prev, desc_next, matches = correspondences.find_correspondences_akaze(im_prev, im_next)

    
    
    
    


    
    


