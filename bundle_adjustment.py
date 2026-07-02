from logger import logger, log_time
from scipy.spatial.transform import Rotation
import numpy as np
import sys
sys.path.append("build/Release")
import cpp_ba

def packet_data(recon):
    camera_params = []
    point_params = []
    observations = []

    for view_id, view in recon.views.items():
        q = Rotation.from_matrix(view.R).as_quat()
        # scipy gives [x,y,z,w]
        camera_params.append([
            float(q[3]),
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(view.t[0]),
            float(view.t[1]),
            float(view.t[2]),
        ])

    for point_id, point in recon.points.items():
        point_params.append([
            float(point.xyz[0]),
            float(point.xyz[1]),
            float(point.xyz[2]),
        ])
    
    # MAPS IN CASE IDs are missing, for example removed point with id 3 gives 0,1,2,4...
    view_ids = list(recon.views.keys())
    point_ids = list(recon.points.keys())

    view_id_to_idx = {
        view_id: i
        for i, view_id in enumerate(view_ids)
    }

    point_id_to_idx = {
        point_id: i
        for i, point_id in enumerate(point_ids)
    }

    for obs in recon.observations.values():
        obs_cpp = cpp_ba.Observation()
        obs_cpp.camera_index = view_id_to_idx[obs.view_id]
        obs_cpp.point_index = point_id_to_idx[obs.point_id]
        obs_cpp.x = float(obs.xy[0])
        obs_cpp.y = float(obs.xy[1])

        observations.append(obs_cpp)
    
    camera_flat = []
    for c in camera_params:
        camera_flat.extend(c)
    point_flat = []
    for p in point_params:
        point_flat.extend(p)
    
    return camera_flat, point_flat, observations

def bundle_adjustment(camera_flat, point_flat, observations):
    results = cpp_ba.bundle_adjustment(
        camera_flat,
        point_flat,
        observations
    )
    return results

def unpack_results(recon, camera_flat, point_flat, observations):
    for i, (view_id, view) in enumerate(recon.views.items()):

        base = i * 7

        qw = camera_flat[base + 0]
        qx = camera_flat[base + 1]
        qy = camera_flat[base + 2]
        qz = camera_flat[base + 3]

        tx = camera_flat[base + 4]
        ty = camera_flat[base + 5]
        tz = camera_flat[base + 6]

        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

        view.R = R
        view.t = np.array([tx, ty, tz])

    for i, (point_id, point) in enumerate(recon.points.items()):

        base = i * 3

        x = point_flat[base + 0]
        y = point_flat[base + 1]
        z = point_flat[base + 2]

        point.xyz = np.array([x, y, z])