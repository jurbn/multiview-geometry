# //
# // Name        : Camara3d_two_views.py
# // Author      : Pablo Azagra, Richard Elvira
# // Version     : V1.0
# // Copyright   : Your copyright notice
# // Description : Codigo a completar para obtener la posición relativa de la camara dentro de dos frames en un video y
# //               y los puntos en el espacio.
# //============================================================================



import cv2 as cv
from scipy.linalg import expm
from GeometricEstimator import *
from Drawer3D import *
import plotly.graph_objs as go
import plotly.io as pio


def nn_dr(matches, ratio=0.75):
    selected_matches = []
    for matchs in matches:
        if matchs[0].distance / matchs[1].distance < ratio:
            selected_matches.append(matchs[0])
    return selected_matches


#Camera Matrix and distortion coefs.
K = np.array([[641.84180665, 0., 311.13895719],
              [0., 641.17105466, 244.65756186],
              [0., 0., 1.]])
dist = np.array([[-0.02331774, 0.25230237, 0., 0., -0.52186379]])


#######DONE: Obtain the images ref_frame and frame1 from the video
cap = cv.VideoCapture('/home/jrb/College/Viscom/Practica3/media/secuencia_a_cam2.avi')  # video capture object
ret, ref_frame = cap.read()
cv.imwrite('1.png', ref_frame)
cap.set(cv.CAP_PROP_POS_FRAMES, 150)
ret, frame1 = cap.read()
cv.imwrite('2.png', frame1)

###### Undistort the images to reduce the reproyection error in the Ransac
ref_frame = cv.undistort(ref_frame, K, dist)
frame1 = cv.undistort(frame1, K, dist)


###### DONE: Extract the keypoints and obtain the good Matches
sift = cv.SIFT_create(500)
bfmatcher = cv.BFMatcher_create(cv.NORM_L2, False)
kp1, des1 = sift.detectAndCompute(ref_frame, None)
kp2, des2 = sift.detectAndCompute(frame1, None)
matches = bfmatcher.knnMatch(des1,  des2, k=2)
good = nn_dr(matches)
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
des = np.float32([des1[m.queryIdx] for m in good])

#img_matches = cv.drawMatches(ref_frame, kp1, frame1, kp2, good, None)
#cv.imshow("Matches", img_matches)
#if cv.waitKey(0) == ord('q'):
#    cv.destroyAllWindows()


#Transform the list of points into float32 homogeneous coord array of shape (3,N)
pts1 = [pts1[i]+(1.,) for i in range(len(pts1))]
pts2 = [pts2[i]+(1.,) for i in range(len(pts2))]
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
pt_arr1 = np.array(pts1)
pt_arr2 = np.array(pts2)
des_arr = np.array(des)
pt_arr1 = pt_arr1.T
pt_arr2 = pt_arr2.T
des_arr = des_arr


#### First camera is set as the origin
p_RS_R_1, q_RS_R_1 = np.array([0.,0.,0.]),np.array([1.,0.,0.,0.])
R_RS_R_1 = quaternion2Matrix(q_RS_R_1)
T_RS_R_1 = np.eye(4)
T_RS_R_1[0:3, 0:3] = R_RS_R_1
T_RS_R_1[0:3, 3:4] = p_RS_R_1.reshape(3, 1)
T_AC1 = T_RS_R_1

###### DONE: From the points estimate the fundamentalMatrix F_21_est. Use function estimateFundamentalMatrixFromMatchesWithRANSAC
pt_arr1 = pt_arr1.reshape((2, pt_arr1.shape[2]))
pt_arr1 = np.vstack((pt_arr1, np.ones((1, pt_arr1.shape[1]))))
pt_arr2 = pt_arr2.reshape((2,pt_arr2.shape[2]))
pt_arr2 = np.vstack((pt_arr2, np.ones((1, pt_arr2.shape[1]))))

F_21_est, inliersMax = estimateFundamentalMatrixFromMatchesWithRANSAC(pt_arr1, pt_arr2)

##### Normalize the solution and filter the points into the inliers
F_21_est = F_21_est / F_21_est[2, 2]
F_12_est = F_21_est.T
des_arr = np.array([des_arr[i] for i in range(len(des_arr)) if inliersMax[i]])
pt_arr2 = np.array([pt_arr2[:, i] for i in range(pt_arr2.shape[1]) if inliersMax[i]])    # se filtran por epipolar
pt_arr1 = np.array([pt_arr1[:, i] for i in range(pt_arr1.shape[1]) if inliersMax[i]])

###### DONE: Obtain the Essential Matrix and estimate the Rotation and translation using the function estimateRotAndTranslFromEssentialMatrix
E = K.T @ F_21_est @ K
t_21_est, RA_21, RB_21 = estimateRotAndTranslFromEssentialMatrix(E)

###### DONE: From the possible solutions obtained use the funcion testEstimatedPosesFromFundamental to obtain the real solution and the triangulation of the 3dPoints X3D
T_21_est, X3D, _, _ = testEstimatedPosesFromFundamental(pt_arr1.T, pt_arr2.T, K, K, RA_21, RB_21, t_21_est.flatten())

##### Plot the 3d Points and the two camera poses (The origin and the second camera pose estimated)
T_12_est = np.linalg.inv(T_21_est)
X3D_w_est = X3D
T_w2_est = T_12_est
mark_est = dict(color='red', size=5)
mark_gt = dict(color='green', size=5)

cap.set(cv.CAP_PROP_POS_FRAMES, 300)
ret, frame3 = cap.read()
cv.imwrite('3.png', frame3)
T_31_est = extractPose(frame3, X3D, des_arr, sift, K, dist)

T_13_est = np.linalg.inv(T_31_est)

fig_triangulation = go.Figure()
drawRefSystem(fig_triangulation, np.eye(4), "W")
drawCamera(fig_triangulation, T_AC1)
drawCamera(fig_triangulation, T_w2_est)
drawCamera(fig_triangulation, T_13_est)
drawPoints(fig_triangulation, X3D_w_est, mark_est)
pio.show(fig_triangulation)
cv.waitKey()
