import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

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

def undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# load all the test images and organize them nicely in a dict of {file_name: image_data}
test_image_fnames = glob.glob('./test_images/straight_lines*.jpg') + glob.glob('./test_images/test*.jpg')

test_images = []
for i in test_image_fnames:
    test_images.append(cv2.imread(i))

test_image_names = []
for i in test_image_fnames:
    test_image_names.append(i.rsplit('\\', 1)[-1])

test_image_dict = zip(test_image_names, test_images)

# time to remove image distortion with data from calibration
undistorted_images = []

for name, image in zip(test_image_names, test_images):
    undistorted = undistort(image, objpoints, imgpoints)
    undistorted_images.append(undistorted)
    cv2.imwrite('./undistorted_test/' + name, undistorted)

# Perspective Transform ---
#   from code determined in perspective_transform.py
src_points = np.float32([[246, 693],[585, 458], [698, 458], [1061, 693]])
dst_points = np.float32([[250, 650], [250, 100], [930, 100], [930, 650]])

def perspective_transform(img, src_points, dst_points):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return transformed

transformed_images = []

for image in undistorted_images:
    transformed_images.append(perspective_transform(image, src_points, dst_points))

for i in range(len((test_image_names))):
    cv2.imwrite('./warped/' + test_image_names[i], transformed_images[i])

# pixel threshold to get the lines
def l_threshold(image, min, max):
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    L = luv[:,:,0]
    binary = np.zeros_like(L)
    binary[(L >= min) & (L <= max)] = 1
    return binary

def b_threshold(image, min, max):
    Lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    b = Lab[:,:,2]
    binary = np.zeros_like(b)
    binary[(b >= min) & (b <= max)] = 1
    return binary

def g_threshold(image, min, max):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray >= min) & (gray <= max)] = 1
    return binary

thresholded_images = []

for image in transformed_images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    l_thresh = l_threshold(image, 220, 255)
    b_thresh = b_threshold(image, 144, 255)
    g_thresh = g_threshold(image, 215, 255)
    combined = np.zeros_like(image)
    combined[(l_thresh == 1) | (b_thresh == 1) | (g_thresh == 1)] = 255
    thresholded_images.append(combined)

for index, image in enumerate(thresholded_images):
    cv2.imwrite('./thresholded/' + test_image_names[index], image)