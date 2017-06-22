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

def threshold(image):
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    l_thresh = l_threshold(image, 210, 255)
    b_thresh = b_threshold(image, 144, 255)
    g_thresh = g_threshold(image, 210, 255)
    combined = np.zeros_like(image)
    combined[(l_thresh == 1) | (b_thresh == 1) | (g_thresh == 1)] = 252
    ret, combined = cv2.threshold(combined, 250, 255, cv2.THRESH_BINARY)
    return combined
	
for image in undistorted_images:
	thresholded_images.append(threshold(image))

# save the thresholded test images
for index, image in enumerate(thresholded_images):
    cv2.imwrite('./thresholded/' + test_image_names[index], image)


# Perspective Transform ---
#   from code determined in perspective_transform.py
src_points = np.float32([[246, 693],[585, 458], [698, 458], [1061, 693]])
dst_points = np.float32([[250, 720], [250, 100], [930, 100], [930, 720]])

M = None

def perspective_transform(img, src_points, dst_points):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return transformed


transformed_images = []

# apply perspective transform to thresholded test images
for image in thresholded_images:
    transformed_images.append(perspective_transform(image, src_points, dst_points))

# save 'warped' images
for i in range(len((test_image_names))):
    cv2.imwrite('./warped/' + test_image_names[i], transformed_images[i])


margin = 100 # How much to slide left and right for searching
window_width = 40

# create a window mask for convolution
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(warped, window_width, window_height, margin, previous_centroids):
    current_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # create nonzero masks in both dimensions
    nonzero = warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_lane_cent_x = []
    right_lane_cent_x = []
    left_lane_inds = []
    right_lane_inds = []

    # First find the two starting positions (unless known from previous frame) for the left and right lane by using
    # np.sum to get the vertical image slice and then np.convolve the vertical image slice with the window template
    if previous_centroids:
        l_center = previous_centroids[0][0]
        r_center = previous_centroids[0][1]
    else:
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        current_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
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
        # if none found, repeat linear trend from last 2 windows
        elif len(left_lane_cent_x) > 1:
            l_center = left_lane_cent_x[level - 2] - left_lane_cent_x[level - 3] + l_center
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        if conv_signal[r_min_index:r_max_index].any():
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # if none found, repeat linear trend from last 2 windows
        elif len(right_lane_cent_x) > 1:
            r_center = right_lane_cent_x[level - 2] - right_lane_cent_x[level - 3] + r_center
        # Add what we found for that layer
        win_y_low = y_pos - (.5 * window_height)
        win_y_high = y_pos + (.5 * window_height)
        win_xleft_low = l_center - margin
        win_xleft_high = l_center + margin
        win_xright_low = r_center - margin
        win_xright_high = r_center + margin
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
        nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
        nonzero_x < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        current_centroids.append((l_center, r_center))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]

    return current_centroids, left_x, left_y, right_x, right_y


def draw_lane(image, previous_centroids, previous_left_fit, previous_right_fit, frames_to_average):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # window settings
    window_height = image.shape[0] / 9  # Break image into 9 vertical layers

    window_centroids, left_x, left_y, right_x, right_y = find_window_centroids(image, window_width, window_height, margin, previous_centroids)
#    print(previous_right_fit, previous_left_fit)

    # If we found any window centers
    if len(window_centroids) > 0:
        left_fit = np.polyfit(left_y, left_x, 2)
        if previous_left_fit.any():
            for i in [0, 1, 2]:
                left_fit[i] = (left_fit[i] + previous_left_fit[i] * frames_to_average) / (1 + frames_to_average)
        right_fit = np.polyfit(right_y, right_x, 2)
        if previous_right_fit.any():
            for i in [0, 1, 2]:
                right_fit[i] = (right_fit[i] + previous_right_fit[i] * frames_to_average) / (1 + frames_to_average)
    # If no window centers found, just display original road image
    else:
        output = np.array(cv2.merge((image, image, image)), np.uint8)

    # plot the fitted polynomials
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_a, left_b, left_c = left_fit[0], left_fit[1], left_fit[2]
    right_a, right_b, right_c = right_fit[0], right_fit[1], right_fit[2]
    a, b, c = (left_a + right_a) / 2, (left_b + right_b) / 2, (left_c + right_c) / 2
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    y_val = np.max(ploty)

    # define conversions in x and y from pixels
    ym_per_pix = 31 / 720 # meters per pixel in y dimension
    xm_per_pix = 3.7 / 710 # meters per pixel in x dimension

    # fit new polynomials to x, y in word space
    left_fit_cr = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_val * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_val * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print('left curve radius:', left_curverad, 'm\nright curve radius:', right_curverad, 'm')
    avg_curve_rad = int((left_curverad + right_curverad) / 2)

    # make blank like image for drawing
    image_zeros = np.zeros_like(image).astype(np.uint8)
    rgb_warped_zeros = np.dstack((image_zeros, image_zeros, image_zeros))

    # make points usable by cv2
    left_fit_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_fit_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left_fit_points, right_fit_points))

    # draw the lane
    cv2.fillPoly(rgb_warped_zeros, np.int32([points]), (0, 255, 0))
    rgb_warped_zeros = perspective_transform(rgb_warped_zeros, dst_points, src_points)
    cv2.putText(rgb_warped_zeros, 'radius of curvature:' + str(avg_curve_rad) + 'm', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return rgb_warped_zeros, window_centroids, left_fit, right_fit, avg_curve_rad

warped = []
original = []

for i in range(len((test_image_names))):
    warped.append(cv2.imread('./warped/' + test_image_names[i]))
    original.append(cv2.imread('./test_images/' + test_image_names[i]))

for i, image in enumerate(warped):
    lane = draw_lane(image)
    composed = cv2.addWeighted(original[i], 1, lane, .5, 1)
    cv2.imwrite('./composed/' + test_image_names[i], composed)


class MyVideoProcessor(object):
    def __init__(self):
        self.last_centroids = []
        self.past_frames_left = []
        self.past_frames_right = []

        self.best_fit_left = np.asarray([])
        self.best_fit_right = np.asarray([])
        self.running_average = 3

    def pipeline_function(self, frame):
        # your lane detection pipeline
        undistorted = undistort(frame, objpoints, imgpoints)
        thresholded = threshold(undistorted)
        transformed = perspective_transform(thresholded, src_points, dst_points)
        lane, self.last_centroids, self.best_fit_left, self.best_fit_right, curve_rad = draw_lane(transformed, self.last_centroids, self.best_fit_left, self.best_fit_right, self.running_average)
        composed = cv2.addWeighted(frame, 1, lane, .5, 1)
        return composed

video_processor_1, video_processor_2, video_processor_3 = MyVideoProcessor(), MyVideoProcessor(), MyVideoProcessor()

from moviepy.editor import VideoFileClip


project_video_output = './output_images/project_video_output.mp4'
clip1 = VideoFileClip('project_video.mp4')
pv_clip = clip1.fl_image(video_processor_1.pipeline_function)
pv_clip.write_videofile(project_video_output, audio=False)

'''
project_video_output = './output_images/challenge_video_output.mp4'
clip1 = VideoFileClip('challenge_video.mp4')
pv_clip = clip1.fl_image(video_processor_2.pipeline_function)
pv_clip.write_videofile(project_video_output, audio=False)


project_video_output = './output_images/harder_challenge_video_output.mp4'
clip1 = VideoFileClip('harder_challenge_video.mp4')
pv_clip = clip1.fl_image(video_processor_3.pipeline_function)
pv_clip.write_videofile(project_video_output, audio=False)
'''