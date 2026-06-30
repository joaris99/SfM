import cv2
import numpy as np
from logger import logger

def find_correspondences_akaze(im1, im2, akaze):
    logger.info("finding akaze correspondences")
    
    keypoints1, descriptors1 = akaze.detectAndCompute(im1, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(im2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # KNN matching for Lowe ratio test
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Keep only good matches
    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Optional: sort by descriptor distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Transform matched points to 2xN matrices
    putative1 = np.array(
        [keypoints1[m.queryIdx].pt for m in good_matches]
    )

    putative2 = np.array(
        [keypoints2[m.trainIdx].pt for m in good_matches]
    )

    logger.info(f"shape of putative1 is {putative1.shape}, shape of putative2 is {putative2.shape}")
    return putative1, putative2