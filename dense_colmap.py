from dataclasses import dataclass
import numpy as np
import open3d as o3d
import numpy as np


@dataclass
class DensePoint3D:
    id: int
    xyz: np.ndarray
    color: np.ndarray

class DenseReconstruction:

    def __init__(self):
        self.points = {}

    def add_point(self, xyz, color):

        point_id = len(self.points)

        self.points[point_id] = DensePoint3D(
            id=point_id,
            xyz=xyz,
            color=color
        )

        return point_id



def load_dense_reconstruction(path):

    dense = DenseReconstruction()

    pcd = o3d.io.read_point_cloud(path)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    for xyz, color in zip(points, colors):

        dense.add_point(
            xyz.astype(np.float64),
            color.astype(np.float64)
        )

    return dense

dense_recon = load_dense_reconstruction(
    "COLMAP_datasets/my/dense/fused.ply"
)
