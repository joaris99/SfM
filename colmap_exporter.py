import os
import shutil
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


class ColmapExporter:

    def __init__(
        self,
        reconstruction,
        K,
        image_width,
        image_height
    ):
        self.reconstruction = reconstruction

        self.K = K
        self.width = image_width
        self.height = image_height
        self.obs_to_feature_idx = {}


    def export(self, output_dir):

        output_dir = os.path.join("COLMAP_datasets", output_dir)

        image_dir = os.path.join(output_dir, "images")

        sparse_dir = os.path.join(output_dir, "sparse", "0")

        os.makedirs(
            image_dir,
            exist_ok=True
        )

        os.makedirs(
            sparse_dir,
            exist_ok=True
        )


        self.export_images(
            image_dir
        )

        self.write_cameras(
            os.path.join(
                sparse_dir,
                "cameras.txt"
            )
        )

        self.write_images(
            os.path.join(
                sparse_dir,
                "images.txt"
            )
        )

        self.write_points3D(
            os.path.join(
                sparse_dir,
                "points3D.txt"
            )
        )


    def export_images(self, image_dir):

        """
        Copy images into COLMAP images folder.
        """

        for view in sorted(
            self.reconstruction.views.values(),
            key=lambda v: v.id
        ):

            filename = (
                f"{view.id:06d}.png"
            )

            path = os.path.join(
                image_dir,
                filename
            )

            img = view.image

            if img.dtype != np.uint8:
                img = (
                    np.clip(img,0,1)
                    *255
                ).astype(np.uint8)

            Image.fromarray(
                img
            ).save(path)



    def write_cameras(self, filename):

        """
        COLMAP camera format:

        CAMERA_ID MODEL WIDTH HEIGHT PARAMS
        """

        fx = self.K[0,0]
        fy = self.K[1,1]
        cx = self.K[0,2]
        cy = self.K[1,2]


        with open(
            filename,
            "w"
        ) as f:

            f.write(
                "# Camera list\n"
            )

            f.write(
                "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"
            )

            f.write(
                f"1 PINHOLE "
                f"{self.width} "
                f"{self.height} "
                f"{fx} {fy} {cx} {cy}\n"
            )



    def write_images(self, filename):

        """
        COLMAP images format:

        IMAGE_ID
        QW QX QY QZ
        TX TY TZ
        CAMERA_ID
        IMAGE_NAME

        followed by:
        POINTS2D_X POINTS2D_Y POINT3D_ID
        """


        with open(
            filename,
            "w"
        ) as f:


            f.write(
                "# Image list\n"
            )


            for view in sorted(
                self.reconstruction.views.values(),
                key=lambda v: v.id
            ):

                # COLMAP expects world -> camera rotation
                R = view.R

                q = Rotation.from_matrix(
                    R
                ).as_quat()

                qx,qy,qz,qw = q


                f.write(
                    f"{view.id} "
                    f"{qw} "
                    f"{qx} "
                    f"{qy} "
                    f"{qz} "
                    f"{view.t[0]} "
                    f"{view.t[1]} "
                    f"{view.t[2]} "
                    f"1 "
                    f"{view.id:06d}.png\n"
                )

                observations = sorted(view.observation_ids)

                points = []

                for feature_idx, obs_id in enumerate(observations):

                    obs = self.reconstruction.observations[obs_id]

                    self.obs_to_feature_idx[(view.id, obs_id)] = feature_idx

                    uv = obs.xy

                    points.append(
                        (
                            uv[0],
                            uv[1],
                            obs.point_id
                        )
                    )


                f.write(
                    " ".join(
                        [
                            f"{x} {y} {pid}"
                            for x,y,pid in points
                        ]
                    )
                )

                f.write(
                    "\n\n"
                )



    def write_points3D(self, filename):

        """
        COLMAP points format:

        POINT3D_ID X Y Z R G B ERROR TRACK[]
        """


        with open(
            filename,
            "w"
        ) as f:


            f.write(
                "# 3D point list\n"
            )


            for point in sorted(
                self.reconstruction.points.values(),
                key=lambda p: p.id
            ):



                xyz = point.xyz

                rgb = (
                    np.clip(
                        self.get_color(point),
                        0,
                        1
                    ) * 255
                ).astype(int)



                track=[]


                # Find observations belonging
                # to this point.
                #
                # Assumes Gaussian order follows
                # Reconstruction point order.
                #
                # Better: store point_id inside Gaussian.

                for obs_id in point.observation_ids:

                    obs = self.reconstruction.observations[obs_id]

                    track.append(
                        (
                            obs.view_id,
                            self.obs_to_feature_idx[(obs.view_id, obs_id)]
                        )
                    )


                track_string = " ".join(
                    [
                        f"{image_id} {point2d_id}"
                        for image_id,point2d_id
                        in track
                    ]
                )


                f.write(
                    f"{point.id} "
                    f"{xyz[0]} "
                    f"{xyz[1]} "
                    f"{xyz[2]} "
                    f"{rgb[0]} "
                    f"{rgb[1]} "
                    f"{rgb[2]} "
                    f"1.0 "
                    f"{track_string}\n"
                )



    def normalized_to_pixel(self, xy):

        """
        Convert:
            x_norm,y_norm

        back to:
            pixel coordinates
        """

        fx = self.K[0,0]
        fy = self.K[1,1]
        cx = self.K[0,2]
        cy = self.K[1,2]


        u = fx * xy[0] + cx
        v = fy * xy[1] + cy


        return np.array(
            [
                u,
                v
            ]
        )

    def get_color(self, point):

        colors = []

        for obs_id in point.observation_ids:

            obs = self.reconstruction.observations[obs_id]
            view = self.reconstruction.views[obs.view_id]

            u = int(round(obs.xy[0]))
            v = int(round(obs.xy[1]))

            if (
                0 <= u < view.image.shape[1]
                and
                0 <= v < view.image.shape[0]
            ):
                colors.append(view.image[v, u] / 255.0)

        if colors:
            return np.mean(colors, axis=0)

        return np.array([1.0, 1.0, 1.0])