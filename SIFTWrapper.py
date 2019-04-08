import cv2
import numpy as np


class SIFTWrapper:

    def __init__(self, im1, im2):

        self.im1 = im1
        self.im2 = im2

        self.kps = []
        self.descs = []

    def compute_keypoints(self):

        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        kp1, des1 = sift.detectAndCompute(self.im1, None)
        kp2, des2 = sift.detectAndCompute(self.im2, None)

        self.kps.append(kp1)
        self.kps.append(kp2)

        self.descs.append(des1)
        self.descs.append(des2)

        return kp1, des1, kp2, des2

    def compute_matches(self, des1, des2):

        matcher = cv2.BFMatcher()
        return matcher.knnMatch(des1, des2, k=2)

    def compute_best_matches(self, r=0.7):

        kp1, des1, kp2, des2 = self.compute_keypoints()
        matches = self.compute_matches(des1, des2)

        good_matches = []
        for m, n in matches:

            # Compute the ratio between best match m, and second best match n here
            if m.distance < r * n.distance:
                good_matches.append(m)

        u1 = []
        u2 = []
        for match in good_matches:
            u1.append(kp1[match.queryIdx].pt)
            u2.append(kp2[match.trainIdx].pt)

        return np.asarray(u1), np.asarray(u2)
