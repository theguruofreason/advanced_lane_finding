import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from advanced_lane_finding import undistort, threshold #perspective_transform, src_points, dst_points, draw_lane
from moviepy.editor import VideoFileClip

test_image = cv2.imread('./test_images/test4.jpg')

# first lets calibrate our camera using the chessboard images
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


null_array = np.array([])

undistorted = undistort(test_image, objpoints, imgpoints)
thresholded, l_thresh, b_thresh, g_thresh, blue_thresh = threshold(undistorted)

l_not_blue = np.zeros_like(l_thresh)
l_not_blue[(l_thresh == 1) & (blue_thresh == 1)] = 255

l_thresh = l_thresh * 255
b_thresh = b_thresh * 255
g_thresh = g_thresh * 255
blue_thresh = blue_thresh * 255

'''
transformed = perspective_transform(thresholded, src_points, dst_points)
lane, _, _, _, _ = draw_lane(transformed, null_array, null_array, null_array, 0)
composed = cv2.addWeighted(image, 1, lane, .5, 1)
'''

cv2.imshow('thresholded', thresholded)
cv2.imshow('l_thresh', l_thresh)
cv2.imshow('b_thresh', b_thresh)
cv2.imshow('g_thresh', g_thresh)
cv2.imshow('l_not_blue', l_not_blue)
cv2.imshow('blue_thresh', blue_thresh)
cv2.waitKey(0)

'''
cv2.imshow('transformed', transformed)
cv2.imshow('composed', composed)
cv2.waitKey(0)
'''