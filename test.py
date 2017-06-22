import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from advanced_lane_finding import undistort, threshold, perspective_transform, src_points, dst_points, draw_lane
from moviepy.editor import VideoFileClip

project_video_output = './output_images/project_video_output.mp4'
clip1 = VideoFileClip('project_video.mp4')

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

image = VideoFileClip.get_frame(clip1, 39.8)
cv2.imwrite('./test_frame.png', image)

null_array = np.array([])

undistorted = undistort(image, objpoints, imgpoints)
thresholded = threshold(undistorted)
transformed = perspective_transform(thresholded, src_points, dst_points)
lane, _, _, _, _ = draw_lane(transformed, null_array, null_array, null_array, 0)
composed = cv2.addWeighted(image, 1, lane, .5, 1)

cv2.imshow('thresholded', thresholded)
cv2.imshow('transformed', transformed)
cv2.imshow('composed', composed)
cv2.waitKey(0)