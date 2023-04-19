"""
CS 6384 Homework 2 Programming
Implement sift_matching() function in this python script
SIFT feature matching
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# TODO: implement this function
# input: des1 is a matrix of SIFT descriptors with shape [m, 128]
# input: des2 is a matrix of SIFT descriptors with shape [n, 128]
# output: index is an array with length m, where the ith element indicates
#         the matched descriptor from des2 for the ith descriptor in des1
# for example, if the 10th element in index is 100, that means des1[10, :] matches to des2[100, :]
# idea: for each descriptor in des1, find its matching by computing L2 distance with all the descriptors in des2
def sift_matching(des1, des2):
    m, _ = des1.shape
    n, _ = des2.shape
    _index = [0] * m
    for i in range(m):
        d1 = des1[i]
        min_idx, min_dist = 0, np.inf
        for j in range(n):
            d2 = des2[j]
            # euclid_dist = np.sqrt(np.sum((d1 - d2) ** 2))
            euclid_dist = np.linalg.norm(d1 - d2)
            if min_dist > euclid_dist:
                min_dist = euclid_dist
                min_idx = j
        _index[i] = min_idx
    return _index


# main function
if __name__ == '__main__':
    # starting time
    start = time.time()

    # read image 1
    rgb_filename1 = 'data/000006-color.jpg'
    im1 = cv2.imread(rgb_filename1)
    width = im1.shape[1]

    # read image 2
    rgb_filename2 = 'data/000007-color.jpg'
    im2 = cv2.imread(rgb_filename2)

    # SIFT feature extractor
    sift = cv2.SIFT_create()

    # detect features on the two images
    # keypoints with the following fields: 'angle', 'class_id', 'convert', 'octave', 'overlap', 'pt', 'response', 'size'
    keypoints_1, descriptors_1 = sift.detectAndCompute(im1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(im2, None)

    # SIFT matching
    # TODO: implement this function
    index = sift_matching(descriptors_1, descriptors_2)

    # visualization for your debugging
    fig = plt.figure()

    # show the concatenated image
    ax = fig.add_subplot(1, 1, 1)
    im = np.concatenate((im1, im2), axis=1)
    plt.imshow(im[:, :, (2, 1, 0)])

    # show feature points
    ax.set_title('SIFT feature matching')
    for i in range(len(keypoints_1)):
        pt = keypoints_1[i].pt
        plt.scatter(x=pt[0], y=pt[1], c='y', s=5)

    for i in range(len(keypoints_2)):
        pt = keypoints_2[i].pt
        plt.scatter(x=pt[0] + width, y=pt[1], c='y', s=5)

        # draw lines to show the matching
    # subsampling by a factor of 10
    for i in range(0, len(keypoints_1), 10):
        pt1 = keypoints_1[i].pt
        matched = index[i]
        pt2 = keypoints_2[matched].pt
        x = [pt1[0], pt2[0] + width]
        y = [pt1[1], pt2[1]]
        plt.plot(x, y, '--', linewidth=1)

    plt.show()

    # end time
    end = time.time()

    # total time taken
    print(f"Runtime of the program is {end - start}")
