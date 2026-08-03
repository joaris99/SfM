import sfm
from tqdm import tqdm
from collections import defaultdict
from romatch import roma_outdoor
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import cv2
import geometry
from scipy.spatial import cKDTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def precompute_roma_pairs(num_images):
    """
    Returns all image pairs (i, j) that will be used for ROMA.

    Each pair is returned once with i < j.
    """
    pairs = set()

    for current_idx in range(2, num_images):
        candidate_views = sfm.select_candidate_views(current_idx)

        for prev_idx in candidate_views[:1]:
            pairs.add((prev_idx, current_idx))

    return sorted(pairs)

def precompute_roma_correspondences(images, num_points=10000):
    roma = roma_outdoor(device=device)
    num_images = len(images)
    pairs = precompute_roma_pairs(num_images)
    roma_matches = defaultdict(dict)
    pil_images = []
    for image in images:
        im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_images.append(im)

    for prev_idx, current_idx in tqdm(pairs, desc="Computing ROMA matches"): 
        
        warp, certainty = roma.match(pil_images[prev_idx], pil_images[current_idx], device=device)
        matches, certainty = roma.sample(warp, certainty, num=num_points)
        H1, W1 = images[prev_idx].shape[:2]
        H2, W2 = images[current_idx].shape[:2]
        kptsA, kptsB = roma.to_pixel_coordinates(matches, H1, W1, H2, W2)
        roma_matches[prev_idx][current_idx] = {"pts1": kptsA.cpu().numpy().astype(np.float32), "pts2": kptsB.cpu().numpy().astype(np.float32), "certainty": certainty.cpu().numpy()}

    return roma_matches

def save_matches(roma_matches, path):
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    for prev_idx in roma_matches:
        for current_idx, data in roma_matches[prev_idx].items():
            np.savez_compressed(save_dir / f"{prev_idx}_{current_idx}.npz", pts1=data["pts1"], pts2=data["pts2"], certainty=data["certainty"])

def load_roma_matches(images, path):
    save_dir = Path(path)

    if save_dir.exists() and any(save_dir.glob("*.npz")):
        print("Loading cached ROMA matches...")
        roma_matches = defaultdict(dict)

        for file in save_dir.glob("*.npz"):
            prev_idx, current_idx = map(int, file.stem.split("_"))
            data = np.load(file)
            roma_matches[prev_idx][current_idx] = {"pts1": data["pts1"], "pts2": data["pts2"], "certainty": data["certainty"]}

        return roma_matches

    print("No cached ROMA matches found. Computing...")

    roma_matches = precompute_roma_correspondences(images)
    save_matches(roma_matches, save_dir)

    return roma_matches

def triangulate_dense_matches(recon, K, view1, view2, match, merger, certainty_threshold=0.7, reproj_threshold=2.0, angle_threshold=5.0):
    """
    Triangulate dense ROMA correspondences between two registered views.
    """

    pts1 = match["pts1"]
    pts2 = match["pts2"]
    certainty = match["certainty"]

    mask = certainty >= certainty_threshold

    if np.count_nonzero(mask) == 0:
        return

    pts1 = pts1[mask]
    pts2 = pts2[mask]

    # Convert to normalized image coordinates
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None,).reshape(-1, 2)

    # Projection matrices WITHOUT intrinsics
    P1 = np.hstack((view1.R, view1.t.reshape(3, 1)))
    P2 = np.hstack((view2.R, view2.t.reshape(3, 1)))

    # Batch triangulation
    points4d = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
    points3d = (points4d[:3] / points4d[3]).T

    # Camera centres
    C1 = -view1.R.T @ view1.t
    C2 = -view2.R.T @ view2.t

    # Camera coordinates
    X_cam1 = (view1.R @ points3d.T).T + view1.t
    X_cam2 = (view2.R @ points3d.T).T + view2.t

    depth_mask = (X_cam1[:, 2] > 0) & (X_cam2[:, 2] > 0)

    v1 = points3d - C1
    v2 = points3d - C2

    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)

    angles = np.degrees(np.arccos(
        np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
    ))

    angle_mask = angles >= angle_threshold

    X_h = np.hstack([points3d, np.ones((len(points3d), 1))])

    proj1 = (K @ (P1 @ X_h.T)).T
    proj2 = (K @ (P2 @ X_h.T)).T

    proj1 = proj1[:, :2] / proj1[:, 2:3]
    proj2 = proj2[:, :2] / proj2[:, 2:3]

    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)

    reproj_mask = np.maximum(err1, err2) <= reproj_threshold


    mask = depth_mask & angle_mask & reproj_mask

    points3d = points3d[mask]
    pts1 = pts1[mask]
    pts2 = pts2[mask]

    point_ids = merger.find_batch(points3d)

    # Indices of points not already in reconstruction
    new_mask = point_ids == -1

    if np.any(new_mask):

        new_points = points3d[new_mask]

        # Merge duplicates inside this batch
        tree = cKDTree(new_points)

        groups = tree.query_ball_tree(tree, merger.merge_radius)

        representative = {}
        keep = np.ones(len(new_points), dtype=bool)

        for i, nbrs in enumerate(groups):
            if i in representative:
                keep[i] = False
                continue

            pid = recon.add_point(new_points[i], -1)

            representative[i] = pid

            for j in nbrs:
                representative[j] = pid

        # Fill point ids
        new_indices = np.flatnonzero(new_mask)

        for local_idx, global_idx in enumerate(new_indices):
            point_ids[global_idx] = representative[local_idx]

        # Rebuild once after all insertions
        merger.rebuild()

    # Add observations
    for pid, p1, p2 in zip(point_ids, pts1, pts2):
        recon.add_observation(p1, view1.id, pid, None)
        recon.add_observation(p2, view2.id, pid, None)


class PointMerger:

    def __init__(self, recon, merge_radius=0.005):
        self.recon = recon
        self.merge_radius = merge_radius

        self.point_ids = np.array(list(recon.points.keys()), dtype=np.int32)

        if len(self.point_ids):
            self.points = np.array(
                [recon.points[i].xyz for i in self.point_ids],
                dtype=np.float64
            )
            self.tree = cKDTree(self.points)
        else:
            self.points = np.empty((0,3))
            self.tree = None

    def find_batch(self, X):
        """
        Returns an array of point ids.
        -1 means the point is not yet in the reconstruction.
        """

        point_ids = np.full(len(X), -1, dtype=np.int32)

        if self.tree is None:
            return point_ids

        dist, idx = self.tree.query(X)

        valid = dist < self.merge_radius
        point_ids[valid] = self.point_ids[idx[valid]]

        return point_ids

    def rebuild(self):
        self.point_ids = np.array(list(self.recon.points.keys()), dtype=np.int32)
        self.points = np.array(
            [self.recon.points[i].xyz for i in self.point_ids],
            dtype=np.float64
        )
        self.tree = cKDTree(self.points)