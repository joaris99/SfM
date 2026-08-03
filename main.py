import numpy as np
import debug
from logger import logger, log_time, log_indent
import data_structure
from tqdm import tqdm
import sfm
import correspondences
import densification

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

# debug.plot_3D(recon)


for iteration in tqdm(range(num_images - 2), desc="Incremental SfM"):
    with log_time(f"iteration {iteration} took"), log_indent():
        current_idx = iteration + 2

        if iteration % 1 == 0:
            sfm.ba(recon, K)
            sfm.remove_high_reprj_err_points(recon, K, threshold=threshold)

        sfm.remove_few_obs_points(recon, iteration) 

            
        # debug.plot_3D(recon)

        object_points = np.empty((0,3))
        image_points = np.empty((0,2))
        correspondence_map = []
        matches = []

        with log_time("select candidate views"):
            im_next = images[current_idx]
            candidate_views = sfm.select_candidate_views(current_idx)
        
        with log_time("find observation/point matches for pnp"):
            for prev_idx in candidate_views:
                prev_view = recon.views[prev_idx]
                result  = sfm.find_correspondences(recon, prev_view, keypoints[prev_idx], keypoints[current_idx], descriptors[prev_idx], descriptors[current_idx])
                object_points = np.vstack((object_points, result["object_points"]))
                image_points = np.vstack((image_points, result["image_points"]))

                for local_idx in range(len(result["object_points"])):
                    correspondence_map.append((len(matches), local_idx))

                matches.append(result)

        with log_time("estimate pose"):
            inliers, R, t = sfm.estimate_pose(K, object_points, image_points)
            view_id = recon.add_view(R, t, im_next)

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

with log_time("finalize sparse reconstruction"), log_indent():
    sfm.finalize_recon(recon, K, num_images, threshold=threshold)

# recon.save("recon_sparse.pkl")

# recon = recon.load("recon_sparse.pkl")
sfm.compute_error(recon, K, verbose=True, mode="sparse")
debug.plot_3D(recon)

with log_time("load ROMA matches"):
    roma_path = "images/my/roma_matches"
    roma_matches = densification.load_roma_matches(images, roma_path)

with log_time("Densification"):
    merger = densification.PointMerger(recon, merge_radius=0.01)
    for i, view1 in tqdm(recon.views.items(), desc="Densification"):
        if i not in roma_matches:
            continue

        for j, match in roma_matches[i].items():
            if j not in recon.views:
                continue

            view2 = recon.views[j]
            densification.triangulate_dense_matches(recon, K, view1, view2, match, merger, reproj_threshold=0.5)

with log_time("finalize dense reconstruction"), log_indent():
    sfm.finalize_dense(recon, K, num_images, threshold=threshold)

sfm.compute_error(recon, K, verbose=True, mode="dense")
debug.plot_3D(recon)

# gaussian_scene = gaussian_data.GaussianScene.from_reconstruction(recon, K)
# exporter = colmap_exporter.ColmapExporter(recon, gaussian_scene, K, 2016, 1512)
# exporter.export("my")