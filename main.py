from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import debug
import correspondences
from logger import logger, log_time, log_indent
import data_structure
import geometry
import open3d as o3d
import bundle_adjustment
import gaussian_data
import colmap_exporter
from tqdm import tqdm

from scipy.spatial.transform import Rotation

import sys
sys.path.append("build/Release")
# cmake --build . --config Release

# K my
K = np.array([[1660.076384971925, 0, 986.3176281647676], 
              [0, 1656.285062933074, 763.9663384634628], 
              [0, 0, 1]])
# high res / 4
# K = np.array([[729.7852678, 0.          , 363.67611205],
#               [0.         , 732.91934794, 456.89364591],
#               [0.         , 0.          , 1.        ]])
distortion_coefficients = np.array([[0.14941547, -0.16006205, -0.03081961, -0.00597472, 0.10109402]])

folder_name = "images/my"
recon = data_structure.Reconstruction()

nr_images = 36

with log_time("load images"):
    images = []
    
    for i in range(36):
        if i < 9:
            im = np.asarray(Image.open(f"{folder_name}/frame0{i + 1}.png"))
        else:
            im = np.asarray(Image.open(f"{folder_name}/frame{i + 1}.png"))
        images.append(im)
    """
    for i in range(nr_images):
        im = np.asarray(ImageOps.exif_transpose(Image.open(f"{folder_name}/frame{i}.jpg")))
        undistorted = cv2.undistort(im, K, distortion_coefficients)
        images.append(undistorted)
    """
    im1 = images[0]
    im2 = images[1]

with log_time("find correspondences"):
    kp1, kp2, desc1, desc2, matches = correspondences.find_correspondences_akaze(im1, im2)
    pts1, pts2, idx1, idx2 = correspondences.get_coordinates(kp1, kp2, matches)
# print(len(pts2))
# debug.plot_correspondences(im1, im2, pts1, pts2)
with log_time("five-point algorithm"):
    # 5-point algorithm + RANSAC
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K,  method=cv2.RANSAC, prob=0.9999, threshold=0.5)
    mask_E = mask_E.ravel().astype(bool)
    matches = matches[mask_E]
    pts1 = pts1[mask_E]
    pts2 = pts2[mask_E]
    idx1 = idx1[mask_E]
    idx2 = idx2[mask_E]

with log_time("recover pose"):
    # Recover relative pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    mask_pose = mask_pose.ravel().astype(bool)
    matches = matches[mask_pose]
    pts1 = pts1[mask_pose]
    pts2 = pts2[mask_pose]
    idx1 = idx1[mask_pose]
    idx2 = idx2[mask_pose]
# print(len(pts2))
# debug.plot_correspondences(im1, im2, pts1, pts2)
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

        p1 = pts1[i]
        p2 = pts2[i]
        y1 = pts1_norm[i]
        y2 = pts2_norm[i]

        point = geometry.triangulate_optimal(C1, C2, y1, y2)
        point_id = recon.add_point(point, 0)

        recon.add_observation(xy=p1, view_id=v1_id, point_id=point_id, feature_idx=idx1[i])
        recon.add_observation(xy=p2, view_id=v2_id, point_id=point_id, feature_idx=idx2[i])

with log_indent():
    for iteration in tqdm(range(nr_images - 2), desc="Incremental SfM"):
        logger.info(f"iteration {iteration}:")
        
        with log_time("packet info for BA"):
            camera_flat, point_flat, observations = bundle_adjustment.packet_data(recon)

        with log_time("bundle Adjustment"):
            results = bundle_adjustment.bundle_adjustment(camera_flat, point_flat, observations, K, verbose=False)
            camera_flat = results.cameras
            point_flat = results.points

        with log_time("unpack BA"):
            bundle_adjustment.unpack_results(recon, camera_flat, point_flat, observations)
        
        ### remove bad points ###

        # remove points with only 2 observations after 10 iterations
        points_to_remove = []
        for point in recon.points.values():
            if len(point.observation_ids) == 2 and iteration - point.created_iteration > 10:
                points_to_remove.append(point.id)
        for i in points_to_remove:
            recon.remove_point(i)
        

        #########################

        ### re-triangulate ###

        ######################
        
        # if iteration % 3 == 0:
        #    debug.plot_3D(recon)
        
        im_prev = images[iteration + 1]
        im_next = images[iteration + 2]
        prev_view = recon.views[iteration + 1]

        with log_time("find correspondences"):
            kp_prev, kp_next, desc_prev, desc_next, matches = correspondences.find_correspondences_akaze(im_prev, im_next)
            pts1, pts2, idx1, idx2 = correspondences.get_coordinates(kp_prev, kp_next, matches)
            pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
            pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
            
        with log_time("find observation point matches"):
            object_points = []
            image_points = []
            match_indices = []
            unmatched_indices = []
            for i in range(len(idx1)):
                if idx1[i] in prev_view.feature_to_observation:
                    obs_id = prev_view.feature_to_observation[idx1[i]]
                    point_id = recon.observations[obs_id].point_id

                    object_points.append(recon.points[point_id].xyz)
                    image_points.append(pts2[i])
                    match_indices.append(i)
                else:
                    unmatched_indices.append(i)
        
            object_points = np.asarray(object_points, dtype=np.float32)
            image_points = np.asarray(image_points, dtype=np.float32)

        with log_time("pnp"):
            success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, K, None, reprojectionError=4.0, 
                                                            confidence=0.999, iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                raise RuntimeError("PnP failed")
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)
        
        with log_time("add views and observations"):
            view_id = recon.add_view(R, t, kp_next, desc_next, im_next)
            for j in inliers.ravel():
                i = match_indices[j]      # original match index
                obs_id = prev_view.feature_to_observation[idx1[i]]
                point_id = recon.observations[obs_id].point_id

                recon.add_observation(xy=pts2[i], view_id=view_id, point_id=point_id, feature_idx=idx2[i])
        
        C_prev = np.concat((prev_view.R, prev_view.t.reshape(3,1)), axis = 1)
        C_next = np.concat((recon.views[view_id].R, recon.views[view_id].t.reshape(3,1)), axis = 1)

        threshold = 2
        
        for i in unmatched_indices:
            feat_prev = idx1[i]
            feat_next = idx2[i]

            p1 = pts1[i]
            p2 = pts2[i]
            y1 = pts1_norm[i]
            y2 = pts2_norm[i]

            point = geometry.triangulate_optimal(C_prev, C_next, y1, y2)

            # Reject points behind either camera
            z1 = (C_prev @ np.append(point, 1.0))[2]
            z2 = (C_next @ np.append(point, 1.0))[2]

            if z1 <= 0 or z2 <= 0:
                continue

            # Reject large reprojection error
            err1 = geometry.reprojection_error(K, C_prev, point, p1)
            err2 = geometry.reprojection_error(K, C_next, point, p2)

            if err1 > threshold or err2 > threshold:
                continue

            point_id = recon.add_point(point, iteration)

            recon.add_observation(xy=p1, view_id=prev_view.id, point_id=point_id, feature_idx=feat_prev,)
            recon.add_observation(xy=p2, view_id=view_id, point_id=point_id, feature_idx=feat_next)

points_to_remove = []
for point in recon.points.values():
    if len(point.observation_ids) == 2 and nr_images - point.created_iteration > 1:
            points_to_remove.append(point.id)
for i in points_to_remove:
    recon.remove_point(i)
    
debug.plot_3D(recon)
with log_time("compute reprojection error"):
    errors = []

    for obs in recon.observations.values():
        point = recon.points[obs.point_id]
        view = recon.views[obs.view_id]

        C = np.concatenate((view.R, view.t.reshape(3, 1)), axis=1)

        err = geometry.reprojection_error(
            K,
            C,
            point.xyz,
            obs.xy
        )

        errors.append(err)

    errors = np.asarray(errors)
    logger.info(f"Observations: {len(errors)}")
    logger.info(f"Mean reprojection error:   {errors.mean():.3f} px")
    logger.info(f"Median reprojection error: {np.median(errors):.3f} px")
    logger.info(f"Std:                       {errors.std():.3f} px")
    logger.info(f"Max:                       {errors.max():.3f} px")
    print(f"Observations: {len(errors)}")
    print(f"Mean reprojection error:   {errors.mean():.3f} px")
    print(f"Median reprojection error: {np.median(errors):.3f} px")
    print(f"Std:                       {errors.std():.3f} px")
    print(f"Max:                       {errors.max():.3f} px")

# gaussian_scene = gaussian_data.GaussianScene.from_reconstruction(recon, K)
# exporter = colmap_exporter.ColmapExporter(recon, gaussian_scene, K, 2016, 1512)
# exporter.export("my")