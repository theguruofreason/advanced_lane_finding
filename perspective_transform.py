import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

test_images = []

for i in glob.glob('./undistorted_test/straight_lines*.jpg'):
    test_images.append(cv2.imread(i))


cv2.imwrite('test', test_images[0])


'''
input_imgs = glob.glob('./undistorted_test/*')

for fname in input_imgs:
    img = cv2.imread(fname)
    img_size = img.shape[:2]
    print(img_size)
'''