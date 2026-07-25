import numpy as np
import debug
from logger import logger, log_time, log_indent
import data_structure
from tqdm import tqdm
import sfm
import correspondences

from scipy.spatial.transform import Rotation

import sys
sys.path.append("build/Release")
# cmake --build . --config Release

data_path = "images/my"
recon = data_structure.Reconstruction()
threshold = 5
angle_threshold = 0.5

images, K, distortion = sfm.load_data(data_path)
num_images = len(images)
keypoints = []
descriptors = []

with log_time("compute keypoint/descriptor pairs"):
    for im in images:
        kp, desc = correspondences.find_keypoints(im)
        keypoints.append(kp)
        descriptors.append(desc)

sfm.setup(recon, K, images[0], images[1], keypoints[0], keypoints[1], descriptors[0], descriptors[1], angle_threshold=angle_threshold)

debug.plot_3D(recon)

with log_indent():
    for iteration in tqdm(range(num_images - 2), desc="Incremental SfM"):
        logger.info(f"iteration {iteration}:")
        current_idx = iteration + 2

        if iteration % 10 == 0:
            sfm.ba(recon, K, iteration)
        
        # debug.plot_3D(recon)

        object_points = np.empty((0,3))
        image_points = np.empty((0,2))
        correspondence_map = []
        matches = []

        with log_time("select candidate views"):
            im_next = images[current_idx]
            candidate_views = sfm.select_candidate_views(current_idx)
        
        with log_indent(), log_time("find observation/point matches for pnp"):
            for prev_idx in candidate_views:
                prev_view = recon.views[prev_idx]
                result  = sfm.find_correspondences(recon, prev_view, im_next, keypoints[prev_idx], keypoints[current_idx], descriptors[prev_idx], descriptors[current_idx])
                object_points = np.vstack((object_points, result["object_points"]))
                image_points = np.vstack((image_points, result["image_points"]))

                for local_idx in range(len(result["object_points"])):
                    correspondence_map.append((len(matches), local_idx))

                matches.append(result)

        with log_time("estimate pose"):
            inliers, R, t = sfm.estimate_pose(K, object_points, image_points)
            view_id = recon.add_view(R, t, matches[0]["kp_next"], matches[0]["desc_next"], im_next)

        with log_time("add old observations"):
            for global_idx in inliers.ravel():
                match_id, local_idx = correspondence_map[global_idx]
                result = matches[match_id]
                prev_idx = candidate_views[match_id]
                prev_view = recon.views[prev_idx]
                sfm.add_old_obs(recon, prev_view, view_id, np.array([local_idx]), result["match_indices"], result["idx1"], result["idx2"], result["pts2"])

        with log_time("add new points"):
            for prev_idx, result in zip(candidate_views, matches):
                prev_view = recon.views[prev_idx]
                sfm.triangulate_new_points(recon, K, prev_view, view_id, result["unmatched_indices"], 
                                        result["pts1"], result["pts2"], result["idx1"], result["idx2"], 
                                        iteration, threshold=threshold,  angle_threshold=angle_threshold)

sfm.finalize_recon(recon, num_images)
sfm.compute_error(recon, K, verbose = True)
debug.plot_3D(recon)

# gaussian_scene = gaussian_data.GaussianScene.from_reconstruction(recon, K)
# exporter = colmap_exporter.ColmapExporter(recon, gaussian_scene, K, 2016, 1512)
# exporter.export("my")