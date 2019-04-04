import sys

import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from scipy.optimize import least_squares

from SIFTWrapper import *

# Need this to plot 3d
from mpl_toolkits.mplot3d import axes3d


def triangulate(P0, P1, x1, x2):

    # P0,P1: projection matrices for each of two cameras/images
    # x1,x1: corresponding points in each of two images (If using P that has been scaled by K, then use camera
    # coordinates, otherwise use generalized coordinates)
    A = np.array([[P0[2, 0] * x1[0] - P0[0, 0], P0[2, 1] * x1[0] - P0[0, 1], P0[2, 2] * x1[0] - P0[0, 2], P0[2, 3] * x1[0] - P0[0, 3]],
                  [P0[2, 0] * x1[1] - P0[1, 0], P0[2, 1] * x1[1] - P0[1, 1], P0[2, 2] * x1[1] - P0[1, 2], P0[2, 3] * x1[1] - P0[1, 3]],
                  [P1[2, 0] * x2[0] - P1[0, 0], P1[2, 1] * x2[0] - P1[0, 1], P1[2, 2] * x2[0] - P1[0, 2], P1[2, 3] * x2[0] - P1[0, 3]],
                  [P1[2, 0] * x2[1] - P1[1, 0], P1[2, 1] * x2[1] - P1[1, 1], P1[2, 2] * x2[1] - P1[1, 2], P1[2, 3] * x2[1] - P1[1, 3]]])

    u, s, vt = np.linalg.svd(A)

    return vt[-1]


def intrinsic_cam_mtx(f, cu, cv):

    return np.asarray([[f, 0, cu],
                       [0, f, cv],
                       [0, 0, 1]])


def plot_matches(img1, u1, img2, u2, skip=10):

    h = img1.shape[0]
    w = img1.shape[1]

    fig = plt.figure(figsize=(12, 12))
    I_new = np.zeros((h, 2 * w, 3)).astype(int)
    I_new[:, :w, :] = img1
    I_new[:, w:, :] = img2
    plt.imshow(I_new)
    plt.scatter(u1[::skip, 0], u1[::skip, 1])
    plt.scatter(u2[::skip, 0] + w, u2[::skip, 1])
    [plt.plot([u1[0], u2[0] + w], [u1[1], u2[1]]) for u1, u2 in zip(u1[::skip], u2[::skip])]
    plt.show()


# TODO: switch to piexif
def get_intrinsic_params(img):

    # Get relevant exif data
    exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}

    f_length_35 = int(exif['FocalLengthIn35mmFilm'])
    img = np.asarray(img)
    h, w, d = img.shape

    f_length = round(f_length_35 / 36 * w, 4)
    sensor_size = (w // 2, h // 2)

    # print("focal length:", f_length)
    # print("sensor size:", sensor_size, '\n')

    return f_length, sensor_size


def get_inliers(u1, u2, K):

    # Make homogeneous
    u1 = np.column_stack((u1, np.ones(shape=(u1.shape[0], 1))))
    u2 = np.column_stack((u2, np.ones(shape=(u2.shape[0], 1))))

    K_inv = np.linalg.inv(K)

    x1 = u1 @ K_inv.T
    x2 = u2 @ K_inv.T

    E, inliers = cv2.findEssentialMat(x1[:, :2], x2[:, :2], np.eye(3), method=cv2.RANSAC, threshold=1e-3)

    # Flatten inlier mask for use w/ numy arrays( (n,1) -> (n,) )
    inliers = inliers.ravel().astype(bool)

    n_in, R, t, _ = cv2.recoverPose(E, x1[inliers, :2], x2[inliers, :2], cameraMatrix=K)

    # Can use this to plot inliers
    # im_1 = plt.imread(sys.argv[1])
    # im_2 = plt.imread(sys.argv[2])
    # h, w, d = im_1.shape
    # skip = 2
    # fig = plt.figure(figsize=(12, 12))
    # I_new = np.zeros((h, 2 * w, 3)).astype(int)
    # I_new[:, :w, :] = im_1
    # I_new[:, w:, :] = im_2
    # plt.imshow(I_new)
    # plt.scatter(u1[inliers, 0][::skip], u1[inliers, 1][::skip])
    # plt.scatter(u2[inliers, 0][::skip] + w, u2[inliers, 1][::skip])
    # [plt.plot([u1[0], u2[0] + w], [u1[1], u2[1]]) for u1, u2 in zip(u1[inliers][::skip], u2[inliers][::skip])]
    # plt.show()

    # FIXME:
    return np.hstack((R, t)), x1[inliers, :2], x2[inliers, :2],  u1[inliers, :2], u2[inliers, :2]
    # return np.hstack((R, t)), x1[inliers, :2], x2[inliers, :2]


def plot_3d(pts, ax, color):
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, alpha=1.0)


def get_point_estimates(P0, P1, x1, x2):

    point_estimates = []
    for x1_pt, x2_pt in zip(x1, x2):
        general_point = triangulate(P0, P1, x1_pt, x2_pt)
        general_point /= general_point[3]

        point_estimates.append(general_point[:3])

    return np.asarray(point_estimates)


def compute_keypoints(im):

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp, des = sift.detectAndCompute(im, None)

    return kp, des


def compute_matches(des1, des2):

    matcher = cv2.BFMatcher()
    return matcher.knnMatch(des1, des2, k=2)


def compute_best_matches(kp1, des1, kp2, des2, r):

    matches = compute_matches(des1, des2)

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

    u1 = np.array(u1)
    u2 = np.array(u2)

    return u1, u2


def get_gcp_mask(orig_inliers, new_inliers):

    three_d_mask = np.full(shape=len(orig_inliers), fill_value=False)
    u_mask = np.full(shape=len(new_inliers), fill_value=False)

    for i, new in enumerate(new_inliers):
        for j, orig in enumerate(orig_inliers):

            if np.allclose(orig, new):
                u_mask[i] = True
                three_d_mask[j] = True
                break

    return u_mask, three_d_mask


def estimate_translation(X_gcp, u_gcp, t0, R, P0):
    """
    This function adjusts the translation vector such that the difference between the observed real world coordinates
    X_gcp and the triangulated real world coordinates of u_gcp is minimized.
    """
    # Note: u_gcp is of the form [u1, v1, u2, v2]
    x1, x2 = u_gcp[:, :2], u_gcp[:, 2:]
    t = t0.copy()

    def residuals(t, X_gcp, x1, x2, R, P0):

        t_tmp = t.copy()
        P1 = np.column_stack((R, t_tmp))

        xyz = get_point_estimates(P0, P1, x1, x2)
        return xyz.ravel() - X_gcp.ravel()

    res = least_squares(residuals, t, method='lm', args=(X_gcp, x1, x2, R, P0))
    # print(res)
    return res.x


def main(argv):

    if len(argv) != 3:
        print("usage: Sequential Structure from Motion: <img1> <img2> <img3>")
        sys.exit(1)

    im1 = Image.open(argv[0])

    f, sensor_size = get_intrinsic_params(im1)
    cu, cv = sensor_size

    im_1 = plt.imread(argv[0])
    im_2 = plt.imread(argv[1])
    im_3 = plt.imread(argv[2])

    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    sw = SIFTWrapper(im_1, im_2)
    u1, u2_1 = sw.compute_best_matches(0.7)

    # Note that here we are assuming all pictures came from same camera
    # (A safe assumption for now, because we KNOW they came from the same camera)
    K_cam = intrinsic_cam_mtx(f, cu, cv)
    extrinsic_cam, x1_inliers, x2_1_inliers, u1_inliers, u2_1_inliers = get_inliers(u1, u2_1, K_cam)

    # Note that we are using generalized
    # camera coordinates, so we do not
    # need to multiply by K(For either of P0, P1)
    P_0 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])

    P_1 = extrinsic_cam

    # Estimate points
    point_estimates = get_point_estimates(P_0, P_1, x1_inliers, x2_1_inliers)

    # Compute keypoints for image 3
    kp3, des3 = compute_keypoints(im_3)

    # Find good matches b/w images 2 and 3
    # Note that we relax r to allow for more full matches
    u2_2, u3 = compute_best_matches(sw.kps[1], sw.descs[1], kp3, des3, r=0.75)

    # Estimate pose(using recovered essential matrix) and get inliers.
    # These inliers will be used for point triangulation
    extrinsic_cam2, x2_2_inliers, x3_inliers, u2_2_inliers, u3_inliers = get_inliers(u2_2, u3, K_cam)

    P_1_gen = np.row_stack((P_1, np.asarray([0, 0, 0, 1])))
    P_2_prime_gen = np.row_stack((extrinsic_cam2, np.asarray([0, 0, 0, 1])))

    # Note that the rotation of this pose is correct wrt P1's coordinate system
    # the translation is also correct, up to scale.
    P_2_init = (P_1_gen @ P_2_prime_gen)[:3]

    # Find set of points for which we have 3D estimates
    u_mask, three_d_mask = get_gcp_mask(u2_1_inliers, u2_2_inliers)

    X_gcp = point_estimates[three_d_mask]
    u_gcp = np.column_stack((x2_2_inliers[u_mask], x3_inliers[u_mask]))

    t0 = P_2_init[:, 3]
    R = P_2_init[:, :3]

    # Minimize projection error between observed and predicted 3d coordinates
    # This essentially fixes the scaling on the translation vector
    # Note that we could make this a bit easier / faster by optimizing only the scale
    # of the translation vector
    t_est = estimate_translation(X_gcp, u_gcp, t0, R, P_1)

    P_2 = np.column_stack((R, t_est))

    second_pt_estimates = get_point_estimates(P_1, P_2, x2_2_inliers, x3_inliers)

    u_mask_inv = np.invert(u_mask)
    new_points = second_pt_estimates[u_mask_inv]

    plot_3d(point_estimates, ax, 'r')
    plot_3d(new_points, ax, 'b')
    plt.show()

    # print("point estimate:", second_pt_estimates[0])
    # print("point actual:", X_gcp[0])
    # print("average distance b/w predicted and observed:", np.sum(np.sqrt(np.sum(np.square(second_pt_estimates[u_mask] - X_gcp)))) / len(X_gcp))

    # num_new = len(new_points)
    # print(num_new, "new points were recovered out of", len(second_pt_estimates))


if __name__ == '__main__':
    main(sys.argv[1:])
