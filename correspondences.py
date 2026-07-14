import cv2
import numpy as np
from logger import logger, log_timing

def find_correspondences_akaze(im1, im2):
    akaze = cv2.AKAZE_create(threshold=0.0001)
    
    keypoints1, descriptors1 = akaze.detectAndCompute(im1, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(im2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)


    # sift = cv2.SIFT_create()

    # keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(im2, None)

    # bf = cv2.BFMatcher(cv2.NORM_L2)

    # KNN matching for Lowe ratio test
    matches12 = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches21 = bf.knnMatch(descriptors2, descriptors1, k=2)

    good12 = {}
    for m, n in matches12:
        if m.distance < 0.75 * n.distance:
            # queryIdx -> trainIdx
            good12[(m.queryIdx, m.trainIdx)] = m

    good21 = {}
    for m, n in matches21:
        if m.distance < 0.75 * n.distance:
            # queryIdx=image2, trainIdx=image1
            good21[(m.queryIdx, m.trainIdx)] = m

    

    # Keep only good matches
    good_matches = []

    for (i, j), m in good12.items():
        if (j, i) in good21:
            good_matches.append(m)

    good_matches.sort(key=lambda m: m.distance)
    

    # Optional: sort by descriptor distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = np.array(good_matches, dtype=object)

    return keypoints1, keypoints2, descriptors1, descriptors2, good_matches

def get_coordinates(kps1, kps2, matches):
    pts1, pts2, idx1, idx2 = [], [], [], []

    for m in matches:
        pts1.append(kps1[m.queryIdx].pt)
        pts2.append(kps2[m.trainIdx].pt)
        idx1.append(m.queryIdx)
        idx2.append(m.trainIdx)

    return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32), np.array(idx1), np.array(idx2)
    