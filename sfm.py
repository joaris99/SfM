from logger import log_time
from PIL import Image
import correspondences
import cv2
import numpy as np
import geometry

def setup(recon, data_path, K):  
    with log_time("load images"):
        images = []
        
        for i in range(36):
            if i < 9:
                im = np.asarray(Image.open(f"{data_path}/frame0{i + 1}.png"))
            else:
                im = np.asarray(Image.open(f"{data_path}/frame{i + 1}.png"))
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
        matches, pts1, pts2, idx1, idx2 = matches[mask_E], pts1[mask_E], pts2[mask_E], idx1[mask_E], idx2[mask_E]
        

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
            
            if angle < 0.5:
                continue

            point_id = recon.add_point(point, 0)

            recon.add_observation(xy=pts1[i], view_id=v1_id, point_id=point_id, feature_idx=idx1[i])
            recon.add_observation(xy=pts2[i], view_id=v2_id, point_id=point_id, feature_idx=idx2[i])
    
    return images

def two_view_recon(recon, prev_view, im_prev, im_next, K, iteration):

    
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
        point = geometry.triangulate_optimal(C_prev, C_next, pts1_norm[i], pts2_norm[i])

        # Reject points with to low angle
        angle = geometry.triangulation_angle(point, prev_view.R, prev_view.t.reshape(3,1), recon.views[view_id].R, recon.views[view_id].t.reshape(3,1))

        if angle < 0.5:
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