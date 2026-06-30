from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import debug
import correspondences
from logger import logger



folder_name = "my"

logger.info("loading images")
im1 = np.asarray(Image.open(f"images/{folder_name}/frame01.png"))
im2 = np.asarray(Image.open(f"images/{folder_name}/frame02.png"))

akaze = cv2.AKAZE_create()
putative1, putative2 = correspondences.find_correspondences_akaze(im1, im2, akaze)

debug.plot_correspondences(im1, im2, putative1, putative2, scale=0.5)
