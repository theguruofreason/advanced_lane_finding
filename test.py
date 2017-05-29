import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

# grab images to calibrate
images = glob.glob('./camera_cal/calibration*.jpg')
num_image = 1

# get corners in calibration images
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # if found, add to our lists
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    cv2.imwrite('./calibrated/calibrated' + str(num_image) + '.jpg', img)
    num_image +=1

num_image = 1

for fname in images:
    img = cv2.imread(fname)

    def cal_undistort(img, objpoints, imgpoints):
        # Use cv2.calibrateCamera() and cv2.undistort()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
        img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    undistorted = cal_undistort(img, objpoints, imgpoints)

    cv2.imwrite('./undistorted/undistorted' + str(num_image) + '.jpg', undistorted)
    num_image +=1

