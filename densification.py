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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def precompute_roma_pairs(num_images):
    """
    Returns all image pairs (i, j) that will be used for ROMA.

    Each pair is returned once with i < j.
    """
    pairs = set()

    for current_idx in range(2, num_images):
        candidate_views = sfm.select_candidate_views(current_idx)

        for prev_idx in candidate_views:
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

"""
def triangulate_dense_matches(recon, K, view1, view2, match):

    pts1 = match["pts1"]
    pts2 = match["pts2"]
    certainty = match["certainty"]

    P1 = K @ np.hstack((view1.R, view1.t.reshape(3, 1)))
    P2 = K @ np.hstack((view2.R, view2.t.reshape(3, 1)))

    for p1, p2, conf in zip(pts1, pts2, certainty):

        if conf < 0.7:
            continue

        X = geometry.triangulate(P1, P2, p1, p2)

        # positive depth
        # triangulation angle
        # reprojection error
        # duplicate test

        point_id = recon.add_point(X)

        recon.add_observation(view1.id, point_id, None, p1)
        recon.add_observation(view2.id, point_id, None, p2)
"""

def triangulate_dense_matches(recon, K, view1, view2, match, certainty_threshold=0.7, reproj_threshold=2.0, angle_threshold=5.0):
    """
    Triangulate dense ROMA correspondences between two registered views.
    """

    pts1 = match["pts1"]
    pts2 = match["pts2"]
    certainty = match["certainty"]

    # ------------------------------------------------------------------
    # Confidence filtering
    # ------------------------------------------------------------------

    mask = certainty >= certainty_threshold

    if np.count_nonzero(mask) == 0:
        return

    pts1 = pts1[mask]
    pts2 = pts2[mask]

    # ------------------------------------------------------------------
    # Convert to normalized image coordinates
    # ------------------------------------------------------------------

    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None,).reshape(-1, 2)

    # ------------------------------------------------------------------
    # Projection matrices WITHOUT intrinsics
    # ------------------------------------------------------------------

    P1 = np.hstack((view1.R, view1.t.reshape(3, 1)))
    P2 = np.hstack((view2.R, view2.t.reshape(3, 1)))

    # ------------------------------------------------------------------
    # Batch triangulation
    # ------------------------------------------------------------------

    points4d = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)

    points3d = (points4d[:3] / points4d[3]).T

    # Camera centres
    C1 = -view1.R.T @ view1.t
    C2 = -view2.R.T @ view2.t

    for X, p1, p2 in zip(points3d, pts1, pts2):

        # --------------------------------------------------------------
        # Positive depth
        # --------------------------------------------------------------

        X_cam1 = view1.R @ X + view1.t
        X_cam2 = view2.R @ X + view2.t

        if X_cam1[2] <= 0 or X_cam2[2] <= 0:
            continue

        # --------------------------------------------------------------
        # Triangulation angle
        # --------------------------------------------------------------

        v1 = X - C1
        v2 = X - C2

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

        if angle < angle_threshold:
            continue

        # --------------------------------------------------------------
        # Reprojection error
        # --------------------------------------------------------------

        X_h = np.append(X, 1.0)

        x1_proj = K @ (P1 @ X_h)
        x1_proj = x1_proj[:2] / x1_proj[2]

        x2_proj = K @ (P2 @ X_h)
        x2_proj = x2_proj[:2] / x2_proj[2]

        err1 = np.linalg.norm(x1_proj - p1)
        err2 = np.linalg.norm(x2_proj - p2)

        if max(err1, err2) > reproj_threshold:
            continue

        # --------------------------------------------------------------
        # Add point
        # --------------------------------------------------------------

        point_id = recon.add_point(X, -1)
        recon.add_observation(p1, view1.id, point_id, None)
        recon.add_observation(p2, view2.id, point_id, None)