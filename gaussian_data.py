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
                point,
                K
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
    def normalized_to_pixel(xy, K):

        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        u = fx * xy[0] + cx
        v = fy * xy[1] + cy

        return np.array([u,v])
    
    @staticmethod
    def get_color(
        reconstruction,
        point,
        K
    ):

        colors=[]

        for obs_id in point.observation_ids:

            obs = reconstruction.observations[obs_id]

            view = reconstruction.views[obs.view_id]


            uv = GaussianScene.normalized_to_pixel(
                obs.xy,
                K
            )

            u,v = uv.astype(int)


            if (
                0 <= u < view.image.shape[1]
                and
                0 <= v < view.image.shape[0]
            ):

                colors.append(
                    view.image[v,u]/255.0
                )


        if len(colors)>0:
            return np.mean(colors, axis=0)

        return np.array([1,1,1])