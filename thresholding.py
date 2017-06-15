import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

image_dirs = ['./warped/test1.jpg', './warped/test4.jpg', './warped/test5.jpg']
images = []
for i in image_dirs:
    images.append(cv2.imread(i))

# pixel threshold to get the lines
def r_threshold(image, min, max):
    R = image[:,:,0]
    binary = np.zeros_like(R)
    binary[(R >= min) & (R <= max)] = 1
    return binary

def g_threshold(image, min, max):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray >= min) & (gray <= max)] = 1
    return binary

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

def s_threshold(image, min, max):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S >= min) & (S <= max)] = 1
    return binary

thresholded_images = []

for image in images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_thresh = r_threshold(image, 120, 255)
    l_thresh = l_threshold(image, 215, 255)
    b_thresh = b_threshold(image, 144, 255)
    s_thresh = s_threshold(image, 120, 255)
    g_thresh = g_threshold(image, 215, 255)
    combined = np.zeros_like(image)
    combined[(g_thresh == 1) | (l_thresh ==1) | (b_thresh == 1) | ((r_thresh == 1) & (s_thresh == 1))] = 255
    cv2.imshow('test', combined)
    cv2.waitKey(0)
