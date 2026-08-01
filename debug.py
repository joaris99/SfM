import cv2
import numpy as np
from logger import logger, log_timing
import open3d as o3d

@log_timing
def plot_correspondences(
    im1,
    im2,
    pts1,
    pts2,
    radius=5,
    circle_thickness=2,
    line_thickness=1,
    scale=1.0,
    max_points=None,
    window_name="Correspondences",
    seed=0
):
    """
    Plot colored correspondences between two images.

    Fixes RGB/BGR visualization issues by converting
    RGB images to OpenCV's BGR format automatically.
    """

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError("pts1 and pts2 must have same length")

    if max_points is not None:
        pts1 = pts1[:max_points]
        pts2 = pts2[:max_points]

    def prepare_image(im):
        """
        Convert image to proper OpenCV display format.
        """

        # Grayscale
        if len(im.shape) == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        # RGBA -> BGR
        if im.shape[2] == 4:
            return cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)

        # RGB -> BGR
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    im1_vis = prepare_image(im1)
    im2_vis = prepare_image(im2)

    h1, w1 = im1_vis.shape[:2]
    h2, w2 = im2_vis.shape[:2]

    # Side-by-side canvas
    canvas = np.zeros(
        (max(h1, h2), w1 + w2, 3),
        dtype=np.uint8
    )

    canvas[:h1, :w1] = im1_vis
    canvas[:h2, w1:w1+w2] = im2_vis

    rng = np.random.default_rng(seed)

    for (x1, y1), (x2, y2) in zip(pts1, pts2):

        color = tuple(
            int(c) for c in rng.integers(0, 256, size=3)
        )

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)) + w1, int(round(y2)))

        # Hollow circles
        cv2.circle(
            canvas,
            p1,
            radius,
            color,
            circle_thickness
        )

        cv2.circle(
            canvas,
            p2,
            radius,
            color,
            circle_thickness
        )

        # Matching line
        cv2.line(
            canvas,
            p1,
            p2,
            color,
            line_thickness
        )

    if scale != 1.0:
        canvas = cv2.resize(
            canvas,
            None,
            fx=scale,
            fy=scale
        )

    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return canvas

def create_camera_frustum(R, t, scale=0.05, color=[1, 0, 0]):
    """
    Create camera frustum as Open3D LineSet.

    R : (3,3) rotation matrix (world -> camera)
    t : (3,) translation vector
    """

    # Camera center in world coordinates
    C = -R.T @ t

    # Frustum in camera coordinates
    points_cam = np.array([
        [0, 0, 0],
        [-1, -1, 2],
        [ 1, -1, 2],
        [ 1,  1, 2],
        [-1,  1, 2]
    ]) * scale

    # Transform to world coordinates
    points_world = (R.T @ points_cam.T).T + C.reshape(1, 3)

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def plot_3D(recon):
    # -------------------------------
    # Collect points
    # -------------------------------
    points = np.array([recon.points[p].xyz for p in recon.points])

    if len(points) == 0:
        print("No points to display.")
        return

    # -------------------------------
    # Normalize scene for visualization
    # -------------------------------
    center = points.mean(axis=0)

    radius = np.linalg.norm(points - center, axis=1).max()
    radius = max(radius, 1e-8)

    vis_scale = 1.0 / radius

    points_vis = (points - center) * vis_scale

    # -------------------------------
    # Point cloud
    # -------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_vis)

    geometries = [pcd]

    # Camera frustum size relative to normalized scene
    frustum_scale = 0.05

    # -------------------------------
    # Cameras
    # -------------------------------
    for view in recon.views.values():

        R = view.R
        t = view.t

        # Camera center
        C = -R.T @ t

        # Normalize camera center
        C_vis = (C - center) * vis_scale

        # Convert back to t for plotting
        t_vis = -R @ C_vis

        cam = create_camera_frustum(
            R,
            t_vis,
            scale=frustum_scale
        )

        geometries.append(cam)

    o3d.visualization.draw_geometries(geometries)