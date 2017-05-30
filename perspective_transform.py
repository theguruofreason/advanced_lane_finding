import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

test_images = []

for i in glob.glob('./undistorted_test/straight_lines*.jpg'):
    test_images.append(cv2.imread(i))

test1 = test_images[0]
print(test1.shape)
vertices = np.array([[246, 693],[585, 458], [698, 458], [1061, 693]], np.int32)
reshaped_vertices = vertices.reshape((-1, 1, 2))
cv2.polylines(test1, [reshaped_vertices], True, (255, 0, 0))


src_points = np.float32(vertices)
dst_points = np.float32([[250, 650], [250, 100], [930, 100], [930, 650]])

def perspective_transform(img, src_points, dst_points):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return transformed

# may choose not to use this, instead reversing arguments to 'perspective_transform'
def rev_perspective_transform(img, src_points, dst_points):
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    rev_transformed = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return rev_transformed

test_warped = perspective_transform(test1, src_points, dst_points)

cv2.imshow('warped', test_warped)
cv2.waitKey(0)


'''
input_imgs = glob.glob('./undistorted_test/*')

for fname in input_imgs:
    img = cv2.imread(fname)
    img_size = img.shape[:2]
    print(img_size)
'''