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
import sfm

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
data_path = "images/my"
recon = data_structure.Reconstruction()
nr_images = 36

images = sfm.setup(recon, data_path, K)

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
        with log_time("remove old points with 2 observations"):
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
        
        prev_view = recon.views[iteration + 1]
        im_prev = images[iteration + 1]
        im_next = images[iteration + 2]
        
        
        sfm.two_view_recon(recon, prev_view, im_prev, im_next, K, iteration)

with log_time("remove points with 2 observations"):
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
    logger.info(f"Observations:              {len(errors):.3f}")
    logger.info(f"Points:                    {len(recon.points):.3f}")
    logger.info(f"Mean reprojection error:   {errors.mean():.3f} px")
    logger.info(f"Median reprojection error: {np.median(errors):.3f} px")
    logger.info(f"Std:                       {errors.std():.3f} px")
    logger.info(f"Max:                       {errors.max():.3f} px")
    print(f"Observations:              {len(errors):.3f}")
    print(f"Points:                    {len(recon.points):.3f}")
    print(f"Mean reprojection error:   {errors.mean():.3f} px")
    print(f"Median reprojection error: {np.median(errors):.3f} px")
    print(f"Std:                       {errors.std():.3f} px")
    print(f"Max:                       {errors.max():.3f} px")

# gaussian_scene = gaussian_data.GaussianScene.from_reconstruction(recon, K)
# exporter = colmap_exporter.ColmapExporter(recon, gaussian_scene, K, 2016, 1512)
# exporter.export("my")