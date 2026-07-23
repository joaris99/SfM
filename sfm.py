from logger import log_time, logger
from PIL import Image, ImageOps
import correspondences
import cv2
import numpy as np
import geometry
import bundle_adjustment
import debug
from pathlib import Path

def load_data(data_path):
    with log_time("load images"):

        folder = Path(data_path)
        num_images = len(list(folder.glob("frame*.jpg")))

        with open(f"{data_path}/camera.txt", "r") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]

        K = np.array([list(map(float, lines[i].split())) for i in range(3)])
        distortion_coefficients = np.array(list(map(float, lines[3].split())))

        images = []
        """
        for i in range(num_images):
            if i < 9:
                im = np.asarray(Image.open(f"{data_path}/frame0{i + 1}.png"))
            else:
                im = np.asarray(Image.open(f"{data_path}/frame{i + 1}.png"))
            images.append(im)
        """
        for i in range(num_images):
            im = np.asarray(ImageOps.exif_transpose(Image.open(f"{data_path}/frame{i}.jpg")))
            undistorted = cv2.undistort(im, K, distortion_coefficients)
            images.append(undistorted)
        

        
    return images, K, distortion_coefficients


def setup(recon, im1, im2, K, angle_threshold = 0.5): 
    """
    initiates SfM with the first 2 views
    """ 

    with log_time("find correspondences"):
        kp1, kp2, desc1, desc2, matches = correspondences.find_correspondences_akaze(im1, im2)
        pts1, pts2, idx1, idx2 = correspondences.get_coordinates(kp1, kp2, matches)
    # print(len(pts2))
    # debug.plot_correspondences(im1, im2, pts1, pts2)
    with log_time("five-point algorithm"):
        # 5-point algorithm + RANSAC
        E, mask_E = cv2.findEssentialMat(pts1, pts2, K,  method=cv2.RANSAC, prob=0.9999, threshold=0.5)
        mask_E = mask_E.ravel().astype(bool)
        matches, pts1, pts2, idx1, idx2 = matches[mask_E], pts1[mask_E], pts2[mask_E], idx1[mask_E], idx2[mask_E]
    # debug.plot_correspondences(im1, im2, pts1, pts2)

    with log_time("recover pose"):
        # Recover relative pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
        mask_pose = mask_pose.ravel().astype(bool)
        matches, pts1, pts2, idx1, idx2 = matches[mask_pose], pts1[mask_pose], pts2[mask_pose], idx1[mask_pose], idx2[mask_pose]
    # print(len(pts2))
    # debug.plot_correspondences(im1, im2, pts1, pts2)
    with log_time("normalize points"):
        # normalize points so that triangulate optimal works
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    with log_time("add initial views, observations and points"):
        C1 = np.concat((np.identity(3), np.array([0, 0, 0]).reshape(3, 1)), axis = 1)
        C2 = np.concat((R, t), axis = 1)
        v1_id = recon.add_view(np.identity(3), np.array([0, 0, 0]), kp1, desc1, im1)
        v2_id = recon.add_view(R, t, kp2, desc2, im2)
        
        for i in range(len(idx1)):
            point = geometry.triangulate_optimal(C1, C2, pts1_norm[i], pts2_norm[i])
            # Reject points with to low angle
            angle = geometry.triangulation_angle(point, np.identity(3), np.array([[0], [0], [0]]), R, t)
            
            if angle < angle_threshold:
                continue

            point_id = recon.add_point(point, 0)

            recon.add_observation(xy=pts1[i], view_id=v1_id, point_id=point_id, feature_idx=idx1[i])
            recon.add_observation(xy=pts2[i], view_id=v2_id, point_id=point_id, feature_idx=idx2[i])


def select_candidate_views(iteration):
    current_idx = iteration + 2

    candidate_views = []
    # Last 5 views
    for offset in range(1, 6):
        idx = current_idx - offset
        if idx >= 0:
            candidate_views.append(idx)

    # 3 evenly spaced older views
    if current_idx > 5:
        older = np.linspace(
            0,
            current_idx - 6,
            min(3, current_idx - 5),
            dtype=int
        )
        candidate_views.extend(older.tolist())

    # Remove duplicates while preserving order
    candidate_views = list(dict.fromkeys(candidate_views))

    return candidate_views

def find_correspondences(recon, prev_view, im_next):
    im_prev = prev_view.image

    with log_time("find correspondences"):
        kp_prev, kp_next, desc_prev, desc_next, matches = correspondences.find_correspondences_akaze(im_prev, im_next)
        pts1, pts2, idx1, idx2 = correspondences.get_coordinates(kp_prev, kp_next, matches)
        
        
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
    
    return {
        "kp_next": kp_next,
        "desc_next": desc_next,

        "pts1": pts1,
        "pts2": pts2,

        "idx1": idx1,
        "idx2": idx2,

        "match_indices": match_indices,
        "unmatched_indices": unmatched_indices,

        "object_points": np.asarray(object_points, dtype=np.float32),
        "image_points": np.asarray(image_points, dtype=np.float32),
    }

def estimate_pose(K, object_points, image_points):
    with log_time("pnp"):
        success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, K, None, reprojectionError=4.0, 
                                                        confidence=0.999, iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            raise RuntimeError("PnP failed")
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

    return inliers, R, t


def add_old_obs(recon, prev_view, view_id, inliers, match_indices, idx1, idx2, pts2):
    new_view = recon.views[view_id]

    for j in inliers.ravel():
        i = match_indices[j]

        obs_id = prev_view.feature_to_observation[idx1[i]]
        point_id = recon.observations[obs_id].point_id

        feature_idx = idx2[i]

        # Already added this feature from another view
        if feature_idx in new_view.feature_to_observation:
            continue

        recon.add_observation(
            xy=pts2[i],
            view_id=view_id,
            point_id=point_id,
            feature_idx=feature_idx
        )


def triangulate_new_points(recon, K, prev_view, view_id, unmatched_indices, pts1, pts2, idx1, idx2, iteration, threshold = 2, angle_threshold = 0.5):

    C_prev = np.concat((prev_view.R, prev_view.t.reshape(3,1)), axis = 1)
    C_next = np.concat((recon.views[view_id].R, recon.views[view_id].t.reshape(3,1)), axis = 1)
    with log_time("normalize points"):
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    
    for i in unmatched_indices:

        # make sure point is not already added
        next_view = recon.views[view_id]
        if idx2[i] in next_view.feature_to_observation:
            continue

        point = geometry.triangulate_optimal(C_prev, C_next, pts1_norm[i], pts2_norm[i])

        # Reject points with to low angle
        angle = geometry.triangulation_angle(point, prev_view.R, prev_view.t.reshape(3,1), recon.views[view_id].R, recon.views[view_id].t.reshape(3,1))

        if angle < angle_threshold:
            continue

        # Reject points behind either camera
        z1 = (C_prev @ np.append(point, 1.0))[2]
        z2 = (C_next @ np.append(point, 1.0))[2]

        if z1 <= 0 or z2 <= 0:
            continue

        # Reject large reprojection error
        err1 = geometry.reprojection_error(K, C_prev, point, pts1[i])
        err2 = geometry.reprojection_error(K, C_next, point, pts2[i])

        if err1 > threshold or err2 > threshold:
            continue

        point_id = recon.add_point(point, iteration)

        feat_prev = idx1[i]
        feat_next = idx2[i]
        recon.add_observation(xy=pts1[i], view_id=prev_view.id, point_id=point_id, feature_idx=feat_prev,)
        recon.add_observation(xy=pts2[i], view_id=view_id, point_id=point_id, feature_idx=feat_next)


def ba(recon, K, iteration):
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


def finalize_recon(recon, num_images):
    with log_time("remove points with 2 observations"):
        points_to_remove = []
        for point in recon.points.values():
            if len(point.observation_ids) == 2 and num_images - point.created_iteration > 1:
                    points_to_remove.append(point.id)
        for i in points_to_remove:
            recon.remove_point(i)


def compute_error(recon, K, verbose = False):
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
        if verbose:
            print(f"Observations:              {len(errors):.3f}")
            print(f"Points:                    {len(recon.points):.3f}")
            print(f"Mean reprojection error:   {errors.mean():.3f} px")
            print(f"Median reprojection error: {np.median(errors):.3f} px")
            print(f"Std:                       {errors.std():.3f} px")
            print(f"Max:                       {errors.max():.3f} px")