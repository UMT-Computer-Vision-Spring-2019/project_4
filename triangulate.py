import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import piexif

I_1 = plt.imread('falcon/DSC03919.JPG')
I_2 = plt.imread('falcon/DSC03920.JPG')

h,w,d = I_1.shape

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(I_1,None)
kp2,des2 = sift.detectAndCompute(I_2,None)

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
#fig = plt.figure(figsize=(12,12))
I_new = np.zeros((h,2*w,3)).astype(int)
I_new[:,:w,:] = I_1
I_new[:,w:,:] = I_2
#plt.imshow(I_new)
##plt.scatter(u1[::skip,0],u1[::skip,1])
#plt.scatter(u2[::skip,0]+w,u2[::skip,1])
#[plt.plot([u1[0],u2[0]+w],[u1[1],u2[1]]) for u1,u2 in zip(u1[::skip],u2[::skip])]
#plt.show()



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
#fig = plt.figure(figsize=(12,12))
I_new = np.zeros((h,2*w,3)).astype(int)
I_new[:,:w,:] = I_1
I_new[:,w:,:] = I_2
#plt.imshow(I_new)
#plt.scatter(u1[inliers,0][::skip],u1[inliers,1][::skip])
#plt.scatter(u2[inliers,0][::skip]+w,u2[inliers,1][::skip])
#[plt.plot([u1[0],u2[0]+w],[u1[1],u2[1]]) for u1,u2 in zip(u1[inliers][::skip],u2[inliers][::skip])]
#plt.show()


n_in,R,t,_ = cv2.recoverPose(E,x1[inliers,:2],x2[inliers,:2])
#print(R,t)


P_1 = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0]])
P_2 = np.hstack((R,t))
#print(P_1,P_2)


P_1c = K_cam @ P_1
P_2c = K_cam @ P_2
#print(P_1c)
#print(P_2c)


def triangulate(P0,P1,x1,x2):
    # P0,P1: projection matrices for each of two cameras/images
    # x1,x1: corresponding points in each of two images (If using P that has been scaled by K, then use camera
    # coordinates, otherwise use generalized coordinates)
    A = np.array([[P0[2,0]*x1[0] - P0[0,0], P0[2,1]*x1[0] - P0[0,1], P0[2,2]*x1[0] - P0[0,2], P0[2,3]*x1[0] - P0[0,3]],
                  [P0[2,0]*x1[1] - P0[1,0], P0[2,1]*x1[1] - P0[1,1], P0[2,2]*x1[1] - P0[1,2], P0[2,3]*x1[1] - P0[1,3]],
                  [P1[2,0]*x2[0] - P1[0,0], P1[2,1]*x2[0] - P1[0,1], P1[2,2]*x2[0] - P1[0,2], P1[2,3]*x2[0] - P1[0,3]],
                  [P1[2,0]*x2[1] - P1[1,0], P1[2,1]*x2[1] - P1[1,1], P1[2,2]*x2[1] - P1[1,2], P1[2,3]*x2[1] - P1[1,3]]])
    u,s,vt = np.linalg.svd(A)
    return vt[-1]



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for point1, point2 in zip(x1, x2):
    vt = triangulate(P_1c, P_2c, point1, point2)
    #print(vt)
    ax.scatter(vt[0], vt[1], zs=vt[2])

plt.show()