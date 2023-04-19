"""
CS 6384 Homework 2 Programming
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# TODO: implement this function
# input: R is a Harris corner score matrix with shape [height, width]
# output: mask with shape [height, width] with values 0 and 1, where 1s indicate corners of the input image
# idea: for each pixel, check its 8 neighborhoods in the image. If the pixel is the maximum compared to these
# 8 neighborhoods, mark it as a corner with value 1. Otherwise, mark it as non-corner with value 0
def non_maximum_suppression(R):
    # my ToDo: Check possibility of Convolution Max Kernel use
    H, W = R.shape
    mask = np.zeros(R.shape)
    dxy = [[-1, -1], [-1, +0], [-1, +1],
           [+0, -1], [+0, +1],
           [+1, -1], [+1, +0], [+1, +1]]
    for i in range(H):
        for j in range(W):
            max = True
            c_max = R[i, j]
            for d in dxy:
                dx, dy = d
                nx = i + dx
                ny = j + dy
                if (0 <= nx < H) and (0 <= ny < W):
                    if c_max <= R[nx, ny]:
                        max = False
                        break
            if max:
                mask[i, j] = 1
    return mask


# TODO: implement this function
# input: im is an RGB image with shape [height, width, 3]
# output: corner_mask with shape [height, width] with values 0 and 1, where 1s indicate corners of the input image
# Follow the steps in Lecture 7 slides 29-30
# You can use opencv functions and numpy functions
def harris_corner(im):

    # step 0: convert RGB to gray-scale image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    H, W = im.shape

    # step 1: compute image gradient using Sobel filters
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    dx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

    # step 2: compute products of derivatives at every pixel
    dx2 = np.square(dx)
    dxy = dx * dy
    dy2 = np.square(dy)

    # step 3: compute the sums of products of derivatives at each pixel using Gaussian filter from OpenCV
    sumx2 = cv2.GaussianBlur(dx2, (3, 3), 0)
    sumxy = cv2.GaussianBlur(dxy, (3, 3), 0)
    sumy2 = cv2.GaussianBlur(dy2, (3, 3), 0)

    matrix_sums = np.zeros([H, W, 2, 2])
    for i in range(H):
        for j in range(W):
            matrix_sums[i, j] = np.array([[sumx2[i, j], sumxy[i, j]],
                                          [sumxy[i, j], sumy2[i, j]]])

    # step 4: compute determinant and trace of the M matrix
    det_m = np.zeros(im.shape)
    trace_m = np.zeros(im.shape)
    for i in range(H):
        for j in range(W):
            det_m[i, j] = np.linalg.det(matrix_sums[i, j])
            trace_m[i, j] = np.trace(matrix_sums[i, j])
    # det_m = np.linalg.det(matrix_sums)
    # trace_m = np.trace(matrix_sums)

    # step 5: compute R scores with k = 0.05
    k = 0.04
    # R = np.zeros(im.shape)
    # for i in range(H):
    #     for j in range(W):
    #         R[i, j] = det_m[i, j] - k * (trace_m[i, j] ** 2)
    R = det_m - k * np.square(trace_m)

    # step 6: thresholding
    # up to now, you shall get an R score matrix with shape [height, width]
    threshold = 0.01 * R.max()
    R[R < threshold] = 0

    # step 7: non-maximum suppression
    # TODO implement the non_maximum_suppression function above
    corner_mask = non_maximum_suppression(R)

    return corner_mask


# main function
if __name__ == '__main__':
    # starting time
    start = time.time()

    # read the image in data
    # rgb image
    rgb_filename = 'data/000006-color.jpg'
    im = cv2.imread(rgb_filename)

    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)

    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()

    # visualization for your debugging
    fig = plt.figure()

    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')

    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('our corner image')

    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('opencv corner image')

    plt.show()

    # end time
    end = time.time()

    # total time taken
    print(f"Runtime of the program is {end - start}")
