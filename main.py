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

for image in undistorted_images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    l_thresh = l_threshold(image, 220, 255)
    b_thresh = b_threshold(image, 144, 255)
    g_thresh = g_threshold(image, 215, 255)
    combined = np.zeros_like(image)
    combined[(l_thresh == 1) | (b_thresh == 1) | (g_thresh == 1)] = 255
    thresholded_images.append(combined)

for index, image in enumerate(thresholded_images):
    cv2.imwrite('./thresholded/' + test_image_names[index], image)

# Perspective Transform ---
#   from code determined in perspective_transform.py
src_points = np.float32([[246, 693],[585, 458], [698, 458], [1061, 693]])
dst_points = np.float32([[250, 650], [250, 100], [930, 100], [930, 650]])

M = None

def perspective_transform(img, src_points, dst_points):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return transformed

transformed_images = []

for image in thresholded_images:
    transformed_images.append(perspective_transform(image, src_points, dst_points))

for i in range(len((test_image_names))):
    cv2.imwrite('./warped/' + test_image_names[i], transformed_images[i])

margin = 80 # How much to slide left and right for searching
window_width = 40

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    plt.imshow(warped)
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    left_lane_cent_x = []
    left_lane_cent_y = []
    right_lane_cent_x = []
    right_lane_cent_y = []

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        print("level:",level, "left lane cent:", left_lane_cent_x)
        y_pos = warped.shape[0] - (level * window_height - .5 * window_height)
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center - offset - margin, 0))
        l_max_index = int(min(l_center - offset + margin, warped.shape[1]))
        if conv_signal[l_min_index:l_max_index].any():
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        elif len(left_lane_cent_x) > 1:
            l_center = left_lane_cent_x[level - 2] - left_lane_cent_x[level - 3] + l_center
        print('left:', l_center, y_pos)
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        if conv_signal[r_min_index:r_max_index].any():
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        elif len(right_lane_cent_x) > 1:
            r_center = right_lane_cent_x[level - 2] - right_lane_cent_x[level - 3] + r_center
        print('right:', r_center, y_pos)
        # Add what we found for that layer
        left_lane_cent_x.append(l_center)
        left_lane_cent_y.append(y_pos)
        right_lane_cent_x.append(r_center)
        right_lane_cent_y.append(y_pos)
        window_centroids.append((l_center, r_center))

    return window_centroids, left_lane_cent_x, left_lane_cent_y, right_lane_cent_x, right_lane_cent_y

def draw_lane(image):
    # window settings
    window_height = image.shape[0] / 9  # Break image into 9 vertical layers

    window_centroids, left_cent_x, left_cent_y, right_cent_x, right_cent_y = find_window_centroids(image, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:
        left_fit = np.polyfit(left_cent_y, left_cent_x, 2)
        right_fit = np.polyfit(right_cent_y, right_cent_x, 2)
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((image, image, image)), np.uint8)

    # plot the fitted polynomials
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # make blank like image for drawing
    image_zeros = np.zeros_like(image).astype(np.uint8)
    rgb_warped_zeros = np.dstack((image_zeros, image_zeros, image_zeros))

    # make centroids usable by cv2
    left_fit_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_fit_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left_fit_points, right_fit_points))

    # draw the lane
    cv2.fillPoly(rgb_warped_zeros, np.int32([points]), (0, 255, 0))
    rgb_warped_zeros = perspective_transform(rgb_warped_zeros, dst_points, src_points)
    return rgb_warped_zeros

warped = []
original = []

for i in range(len((test_image_names))):
    warped.append(cv2.imread('./warped/' + test_image_names[i], 0))
    original.append(cv2.imread('./test_images/' + test_image_names[i]))

for i, image in enumerate(warped):
    lane = draw_lane(image)
    composed = cv2.addWeighted(original[i], 1, lane, .5, 1)
    cv2.imwrite('./composed/' + test_image_names[i], composed)

