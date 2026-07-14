from dataclasses import dataclass, field
import numpy as np
from PIL import Image

@dataclass
class View:
    id: int
    image: np.ndarray
    R: np.ndarray
    t: np.ndarray
    keypoints: list
    descriptors: np.ndarray
    observation_ids: set[int] = field(default_factory=set)
    feature_to_observation: dict[int, int] = field(default_factory=dict)

@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    observation_ids: set[int] = field(default_factory=set)

@dataclass
class Observation:
    id: int
    xy: np.ndarray
    view_id: int
    point_id: int
    feature_idx: int


class Reconstruction:


    def __init__(self):
        self.views = {}
        self.points = {}
        self.observations = {}

        self.next_view_id = 0
        self.next_point_id = 0
        self.next_obs_id = 0
    
    def add_observation(self, xy, view_id, point_id, feature_idx):

        if view_id not in self.views:
            raise ValueError(f"view_id {view_id} does not exist")

        if point_id not in self.points:
            raise ValueError(f"point_id {point_id} does not exist")
        
        if feature_idx in  self.views[view_id].feature_to_observation:
            raise ValueError(f"Feature {feature_idx} already has an observation.")

        obs_id = self.next_obs_id
        self.next_obs_id += 1

        obs = Observation(
            id = obs_id,
            xy = xy,
            view_id = view_id,
            point_id = point_id,
            feature_idx = feature_idx
        )

        self.observations[obs_id] = obs

        self.views[view_id].observation_ids.add(obs_id)
        self.views[view_id].feature_to_observation[feature_idx] = obs_id
        self.points[point_id].observation_ids.add(obs_id)
        return obs_id
    
    def add_view(self, R, t, keypoints, descriptors, image):
        view_id = self.next_view_id
        self.next_view_id += 1

        v = View(
            id = view_id, 
            R = R, 
            t = np.asarray(t, dtype=float).reshape(3,),
            image = image, 
            keypoints=keypoints,
            descriptors=descriptors,
            observation_ids = set()
        )

        self.views[view_id] = v
        return view_id

    def add_point(self, xyz):
        point_id = self.next_point_id
        self.next_point_id += 1

        point = Point3D(
            id = point_id,
            xyz = xyz,
            observation_ids = set()
        )

        self.points[point_id] = point
        return point_id
    
    def remove_observation(self, obs_id):
        obs = self.observations.pop(obs_id)

        self.views[obs.view_id].observation_ids.remove(obs_id)
        del self.view[obs.view_id].feature_to_observation[obs.feature_idx]
        self.points[obs.point_id].observation_ids.remove(obs_id)
    
    def remove_view(self, view_id):
        view = self.views.pop(view_id)

        for obs_id in list(view.observation_ids):
            self.remove_observation(obs_id)
    
    def remove_point(self, point_id):
        point = self.points.pop(point_id)

        for obs_id in list(point.observation_ids):
            self.remove_observation(obs_id)

    
    