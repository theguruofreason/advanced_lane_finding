import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

def calibrate(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # if found, add to our lists
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# grab images to calibrate
calibration_images = glob.glob('./camera_cal/calibration*.jpg')

# get corners in calibration images
for fname in calibration_images:
    image = cv2.imread(fname)
    calibrate(image)

def undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

test_images = glob.glob('./test_images/straight_lines*.jpg') + glob.glob('./test_images/test*.jpg')
undistorted_images = []

for fname in test_images:
    image = cv2.imread(fname)
    undistorted = undistort(image, objpoints, imgpoints)
    undistorted_images.append(undistorted)
    cv2.imwrite('./undistorted_test/' + fname.rsplit('/', 1)[-1], undistorted)
    print('saved as ./undistorted_test/' + fname.rsplit('/', 1)[-1])

cv2.imshow('test', undistorted_images[0])