import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

np.set_printoptions(threshold=np.nan)

image_fnames = glob.glob('/thresholded/*')
images = []
for i in image_fnames:
    images.append(cv2.imread(i))

warped = cv2.imread('./thresholded/test4.jpg', 0)
nonzero = warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# window settings
window_width = 40
window_height = warped.shape[0] / 9 # Break image into 9 vertical layers
print('window height =', window_height)
margin = 120 # How much to slide left and right for searching
left_lane_inds = []
right_lane_inds = []
left_lane_cent_x = []
left_lane_cent_y = []
right_lane_cent_x = []
right_lane_cent_y = []


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

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
        y_pos = warped.shape[0] - (level * window_height - .5 * window_height)
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        if conv_signal[l_min_index:l_max_index].any():
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        print('left:', l_center, y_pos)
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        if conv_signal[r_min_index:r_max_index].any():
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        print('right:', r_center, y_pos)
        # Add what we found for that layer
        left_lane_cent_x.append(l_center)
        left_lane_cent_y.append(y_pos)
        right_lane_cent_x.append(r_center)
        right_lane_cent_y.append(y_pos)
        window_centroids.append((l_center, r_center))

    return window_centroids


window_centroids = find_window_centroids(warped, window_width, window_height, margin)
print(left_lane_cent_y, left_lane_cent_x)
left_centroids = window_centroids[0]
right_centroids = window_centroids[1]

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # Draw the results
    template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
    zero_channel = np.zeros_like(template)  # create a zero color channel
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)  # making the original road pixels 3 color channels
    left_fit = np.polyfit(left_lane_cent_y, left_lane_cent_x, 2)
    right_fit = np.polyfit(right_lane_cent_y, right_lane_cent_x, 2)
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

# plot the fitted polynomials
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# make blank like warped for drawing
warped_zeros = np.zeros_like(warped).astype(np.uint8)
rgb_warped_zeros = np.dstack((warped_zeros, warped_zeros, warped_zeros))

# make centroids usable by cv2
left_fit_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
right_fit_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
points = np.hstack((left_fit_points, right_fit_points))

# draw the lane
cv2.fillPoly(rgb_warped_zeros, np.int_([points]), (0, 255, 0))
output = cv2.addWeighted(output, 1, rgb_warped_zeros, 0.5, 0.0)

# show the result
plt.imshow(output)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, warped.shape[1])
plt.ylim(warped.shape[0], 0)
plt.show()