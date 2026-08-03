import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates
from tqdm import tqdm

def extract_point_cloud(recon, path):

    point_xyz = []
    point_colors = []

    # accumulate colors per point
    color_acc = {}
    color_count = {}

    # group observations by view
    view_observations = {}

    for obs in recon.observations.values():
        view_observations.setdefault(obs.view_id, []).append(obs)


    # process each image once
    for view_id, observations in tqdm(
        view_observations.items(),
        desc="extracting colors"
    ):

        view = recon.views[view_id]
        img = view.image

        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)

        # all pixel coordinates for this view
        xy = np.array([obs.xy for obs in observations])

        xs = xy[:, 0]
        ys = xy[:, 1]

        # bilinear sample all points simultaneously
        sampled = np.stack([
            map_coordinates(
                img[:, :, c],
                [ys, xs],
                order=1,
                mode="nearest"
            )
            for c in range(3)
        ], axis=1)

        sampled = sampled.astype(np.float32) / 255.0


        # assign colors to points
        for obs, color in zip(observations, sampled):

            pid = recon.observations[obs.id].point_id

            if pid not in color_acc:
                color_acc[pid] = color.copy()
                color_count[pid] = 1
            else:
                color_acc[pid] += color
                color_count[pid] += 1



    # build cloud
    for pid, p in recon.points.items():

        if pid not in color_acc:
            continue

        point_xyz.append(p.xyz)

        point_colors.append(
            color_acc[pid] / color_count[pid]
        )


    points = np.asarray(point_xyz)
    colors = np.asarray(point_colors)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(
        f"{path}.ply",
        pcd
    )

def reconstruction(path):
    pcd = o3d.io.read_point_cloud(f"{path}.ply")

    bbox = pcd.get_axis_aligned_bounding_box()
    size = bbox.get_extent()
    radius = np.linalg.norm(size) * 0.01

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30
        )
    )

    pcd.orient_normals_consistent_tangent_plane(
        50
    )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=10,
        width=0,
        scale=1.1
    )

    densities = np.asarray(densities)

    threshold = np.quantile(densities, 0.05)

    vertices_to_remove = densities < threshold

    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(
        f"{path}_poisson_mesh.ply",
        mesh
    )


    