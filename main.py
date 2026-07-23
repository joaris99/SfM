from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import debug
import correspondences
from logger import logger, log_time, log_indent
import data_structure
import geometry
import open3d as o3d
import bundle_adjustment
import gaussian_data
import colmap_exporter
from tqdm import tqdm
import sfm

from scipy.spatial.transform import Rotation

import sys
sys.path.append("build/Release")
# cmake --build . --config Release

data_path = "images/my"
recon = data_structure.Reconstruction()

images, K, distortion = sfm.load_data(data_path)
num_images = len(images)
sfm.setup(recon, images[0], images[1], K)

with log_indent():
    for iteration in tqdm(range(num_images - 2), desc="Incremental SfM"):
        logger.info(f"iteration {iteration}:")
        sfm.ba(recon, K, iteration)

        prev_view = recon.views[iteration + 1]
        im_prev = images[iteration + 1]
        im_next = images[iteration + 2]
        
        sfm.two_view_recon(recon, prev_view, im_prev, im_next, K, iteration)

sfm.finalize_recon(recon, num_images)
sfm.compute_error(recon, K, verbose = True)
debug.plot_3D(recon)

# gaussian_scene = gaussian_data.GaussianScene.from_reconstruction(recon, K)
# exporter = colmap_exporter.ColmapExporter(recon, gaussian_scene, K, 2016, 1512)
# exporter.export("my")