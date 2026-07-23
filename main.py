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

data_path = "images/my"
recon = data_structure.Reconstruction()
threshold = 5
angle_threshold = 0.5

images, K, distortion = sfm.load_data(data_path)
print(K, distortion)
num_images = len(images)
sfm.setup(recon, images[0], images[1], K, angle_threshold=angle_threshold)

debug.plot_3D(recon)

with log_indent():
    for iteration in tqdm(range(num_images - 2), desc="Incremental SfM"):
        logger.info(f"iteration {iteration}:")
        sfm.ba(recon, K, iteration)
        
        # debug.plot_3D(recon)

        im_next = images[iteration + 2]
        candidate_views = sfm.select_candidate_views(iteration)
        object_points = np.empty((0,3))
        image_points = np.empty((0,2))
        correspondence_map = []
        matches = []

        for prev_idx in candidate_views:
            prev_view = recon.views[prev_idx]
            result  = sfm.find_correspondences(recon, prev_view, im_next)
            object_points = np.vstack((object_points, result["object_points"]))
            image_points = np.vstack((image_points, result["image_points"]))

            for local_idx in range(len(result["object_points"])):
                correspondence_map.append((len(matches), local_idx))

            matches.append(result)

        inliers, R, t = sfm.estimate_pose(K, object_points, image_points)
        view_id = recon.add_view(R, t, matches[0]["kp_next"], matches[0]["desc_next"], im_next)

        with log_time("add old observations"):
            for global_idx in inliers.ravel():
                match_id, local_idx = correspondence_map[global_idx]
                result = matches[match_id]
                prev_idx = candidate_views[match_id]
                prev_view = recon.views[prev_idx]
                sfm.add_old_obs(recon, prev_view, view_id, np.array([local_idx]), result["match_indices"], result["idx1"], result["idx2"], result["pts2"])

        for prev_idx, result in zip(candidate_views, matches):
            prev_view = recon.views[prev_idx]
            sfm.triangulate_new_points(recon, K, prev_view, view_id, result["unmatched_indices"], result["pts1"], result["pts2"], result["idx1"], result["idx2"], iteration, threshold=threshold,  angle_threshold=angle_threshold)

sfm.finalize_recon(recon, num_images)
sfm.compute_error(recon, K, verbose = True)
debug.plot_3D(recon)

# gaussian_scene = gaussian_data.GaussianScene.from_reconstruction(recon, K)
# exporter = colmap_exporter.ColmapExporter(recon, gaussian_scene, K, 2016, 1512)
# exporter.export("my")