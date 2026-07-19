from dataclasses import dataclass
import numpy as np


@dataclass
class Gaussian:
    xyz: np.ndarray
    color: np.ndarray
    opacity: float
    scale: np.ndarray
    rotation: np.ndarray

class GaussianScene:


    def __init__(self):
        self.gaussians = []


    @classmethod
    def from_reconstruction(
        cls,
        reconstruction,
        K
    ):

        scene = cls()

        for point in reconstruction.points.values():

            color = cls.get_color(
                reconstruction,
                point
            )


            gaussian = Gaussian(
                xyz=point.xyz,

                color=color,

                opacity=0.5,

                scale=np.array(
                    [0.01,0.01,0.01]
                ),

                rotation=np.array(
                    [1,0,0,0]
                )
            )

            scene.gaussians.append(
                gaussian
            )

        return scene
    
    @staticmethod
    def get_color(
        reconstruction,
        point
    ):

        colors = []

        for obs_id in point.observation_ids:

            obs = reconstruction.observations[obs_id]

            view = reconstruction.views[obs.view_id]

            u, v = obs.xy.astype(int)

            if (
                0 <= u < view.image.shape[1]
                and
                0 <= v < view.image.shape[0]
            ):

                colors.append(
                    view.image[v, u] / 255.0
                )

        if len(colors) > 0:
            return np.mean(colors, axis=0)

        return np.array([1,1,1])