import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import piexif
from scipy.optimize import least_squares
import sys


def estimate_pose(pair1, pair2, cidx1, cidx2, X1, P3_est):
    R = P3_est[:, :-1]
    t0 = P3_est[:, -1].reshape((3, 1))
    P2 = pair1.P_2
    P2c = pair1.K @ P2

    targets = X1[cidx1]
    r0 = cv2.Rodrigues(R)[0]
    p0 = list(r0.ravel())
    p0.extend(t0.ravel())
    u1 = pair2.u1[cidx2]
    u2 = pair2.u2[cidx2]

    def residuals(p):
        R = cv2.Rodrigues(p[:3])[0]
        t = p[3:].reshape((3, 1))
        P3 = np.hstack((R, t))
        P3c = pair2.K @ P3
        Xest = triangulate(P2c, P3c, u1, u2)
        return targets.ravel() - Xest.ravel()

    res = least_squares(residuals, p0)
    p = res.x
    R = cv2.Rodrigues(p[:3])[0]
    t = p[3:].reshape((3, 1))
    P = np.hstack((R, t))
    return P

def loadImages(imageURLs):
    imgs = []
    for URL in imageURLs:
        imgs.append(plt.imread(URL))
    return imgs

def getDesc(img):
    sift = cv2.xfeatures2d.SIFT_create()
    desc,kp = sift.detectAndCompute(img,None)
    return desc,kp

def getMatches(kp1, kp2):

    bf = cv2.BFMatcher()
    matches = []
    goodMatches = []
    matches = (bf.knnMatch(kp1,kp2,k=2))

    for j,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            goodMatches.append(m)
    return goodMatches

def ransacPose(I_1, I_2, good, kp1, kp2):

    u1 = []
    u2 = []

    h,w,d = I_1.shape

    for m in good:
        u1.append(kp1[m.queryIdx].pt)
        u2.append(kp2[m.trainIdx].pt)

    u1 = np.array(u1)
    u2 = np.array(u2)

    #Make homogeneous
    u1 = np.c_[u1,np.ones(u1.shape[0])]
    u2 = np.c_[u2,np.ones(u2.shape[0])]

    skip = 1

    I_new = np.zeros((h,2*w,3)).astype(int)
    I_new[:,:w,:] = I_1
    I_new[:,w:,:] = I_2


    h,w,d = I_1.shape
    exif = piexif.load('falcon/DSC03919.JPG')
    f = exif['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm]/36*w
    cu = w//2
    cv = h//2

    K_cam = np.array([[f,0,cu],[0,f,cv],[0,0,1]])
    K_inv = np.linalg.inv(K_cam)
    x1 = u1 @ K_inv.T
    x2 = u2 @ K_inv.T
    #print(x1)


    E,inliers = cv2.findEssentialMat(x1[:,:2],x2[:,:2],np.eye(3),method=cv2.RANSAC,threshold=1e-3)
    inliers = inliers.ravel().astype(bool)
    #print(E,inliers)


    skip = 10

    I_new = np.zeros((h,2*w,3)).astype(int)
    I_new[:,:w,:] = I_1
    I_new[:,w:,:] = I_2


    n_in,R,t,_ = cv2.recoverPose(E,x1[inliers,:2],x2[inliers,:2])

    P_1 = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0]])
    P_2 = np.hstack((R,t))

    P_1c = K_cam @ P_1
    P_2c = K_cam @ P_2

    return(x1,x2,P_1c,P_2c)

def tester(imageURLs):

    images = loadImages(imageURLs)

    desc_array = []
    KP_array = []
    matches = []
    for img in images:
        desc, kp = getDesc(img)
        desc_array.append(desc)
        KP_array.append(kp)

    for i in range(len(KP_array)-1):
        matches.append(getMatches(KP_array[i],KP_array[i+1]))


    x1,x2,p1,p2 = ransacPose(images[0],images[1], matches[0],desc_array[0],desc_array[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for point1, point2 in zip(x1, x2):
        vt1 = triangulate(p1, p2, point1, point2)
        #vt2 = triangulate(p2, p3, point2, point3)
        ax.scatter(vt1[0], vt1[1], zs=vt1[2])
        #ax.scatter(vt2[0], vt2[1], zs=vt2[2])


    plt.show()
def getPoints(image1URL, image2URL):

    sift = cv2.xfeatures2d.SIFT_create()

    I_1 = plt.imread(image1URL)
    I_2 = plt.imread(image2URL)

    h,w,d = I_1.shape


    kp1,des1 = sift.detectAndCompute(I_1,None)
    kp2,des2 = sift.detectAndCompute(I_2,None)

    h,w,d = I_1.shape

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)

    u1 = []
    u2 = []

    for m in good:

        u1.append(kp1[m.queryIdx].pt)
        u2.append(kp2[m.trainIdx].pt)

    u1 = np.array(u1)
    u2 = np.array(u2)

    #Make homogeneous
    u1 = np.c_[u1,np.ones(u1.shape[0])]
    u2 = np.c_[u2,np.ones(u2.shape[0])]

    skip = 1

    I_new = np.zeros((h,2*w,3)).astype(int)
    I_new[:,:w,:] = I_1
    I_new[:,w:,:] = I_2


    h,w,d = I_1.shape
    exif = piexif.load('falcon/DSC03919.JPG')
    f = exif['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm]/36*w
    cu = w//2
    cv = h//2

    K_cam = np.array([[f,0,cu],[0,f,cv],[0,0,1]])
    K_inv = np.linalg.inv(K_cam)
    x1 = u1 @ K_inv.T
    x2 = u2 @ K_inv.T
    #print(x1)


    E,inliers = cv2.findEssentialMat(x1[:,:2],x2[:,:2],np.eye(3),method=cv2.RANSAC,threshold=1e-3)
    inliers = inliers.ravel().astype(bool)
    #print(E,inliers)


    skip = 10

    I_new = np.zeros((h,2*w,3)).astype(int)
    I_new[:,:w,:] = I_1
    I_new[:,w:,:] = I_2


    n_in,R,t,_ = cv2.recoverPose(E,x1[inliers,:2],x2[inliers,:2])

    P_1 = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0]])
    P_2 = np.hstack((R,t))

    P_1c = K_cam @ P_1
    P_2c = K_cam @ P_2
    #print(P_1c)
    #print(P_2c)


    return(x1,x2,P_1c,P_2c)

def triangulate(P0,P1,x1,x2):
    # P0,P1: projection matrices for each of two cameras/images
    # x1,x2: corresponding points in each of two images (If using P that has been scaled by K, then use camera
    # coordinates, otherwise use generalized coordinates)
    A = np.array([[P0[2,0]*x1[0] - P0[0,0], P0[2,1]*x1[0] - P0[0,1], P0[2,2]*x1[0] - P0[0,2], P0[2,3]*x1[0] - P0[0,3]],
                  [P0[2,0]*x1[1] - P0[1,0], P0[2,1]*x1[1] - P0[1,1], P0[2,2]*x1[1] - P0[1,2], P0[2,3]*x1[1] - P0[1,3]],
                  [P1[2,0]*x2[0] - P1[0,0], P1[2,1]*x2[0] - P1[0,1], P1[2,2]*x2[0] - P1[0,2], P1[2,3]*x2[0] - P1[0,3]],
                  [P1[2,0]*x2[1] - P1[1,0], P1[2,1]*x2[1] - P1[1,1], P1[2,2]*x2[1] - P1[1,2], P1[2,3]*x2[1] - P1[1,3]]])
    u,s,vt = np.linalg.svd(A)
    return vt[-1]


imageURLs = ["falcon/DSC03919.JPG", "falcon/DSC03920.JPG", "falcon/DSC03921.JPG"]
#imageURLs = ["falcon/DSC03919.JPG", "falcon/DSC03920.JPG"]


tester(imageURLs)

#x1,x2,p1,p2 = getPoints("falcon/DSC03919.JPG","falcon/DSC03920.JPG")

#x2p,x3,p2p,p3 = getPoints(sys.argv[2],sys.argv[3])

#print(x2,x2p)

#Append 1s to each row to make them the same shape so the following works.

#p3 = (np.vstack((p2,[0,0,0,1])) @ np.vstack((p3,[0,0,0,1])))[:-1]

#print(p3)


'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for point1, point2 in zip(x1, x2,):
    vt1 = triangulate(p1, p2, point1, point2)
    #vt2 = triangulate(p2, p3, point2, point3)
    ax.scatter(vt1[0], vt1[1], zs=vt1[2])
    #ax.scatter(vt2[0], vt2[1], zs=vt2[2])


plt.show()
'''
