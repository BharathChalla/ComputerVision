"""
CS 6384 Homework 1 Programming
Transform images
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


# transform the input image im (H, W, 3) according to the 2D transformation T (3x3 matrix)
# the output is the transformed image with the same shape (H, W, 3)
# TODO: implementation this function
def transform(im, T):
    H, W, _ = im.shape
    im_new = np.zeros(im.shape, dtype='uint8')
    T_inv = np.linalg.inv(T)

    for i in range(H):
        for j in range(W):
            res = np.dot(T_inv, [j, i, 1]) / np.dot(T_inv[-1, :], [j, i, 1])
            i2, j2, _ = res.astype(int)
            if 0 <= i2 < W:
                if 0 <= j2 < H:
                    im_new[i, j] = im[j2, i2]
    return im_new


# main function
# notice you cannot run this main function until you implement the above transform() function 
if __name__ == '__main__':
    # load the image in data
    filename = 'data/000006-color.jpg'
    im = cv2.imread(filename)

    # image height and width
    height = im.shape[0]
    width = im.shape[1]

    # 2D translation
    T1 = np.eye(3, dtype=np.float32)
    T1[0, 2] = 50
    T1[1, 2] = 100
    im_1 = transform(im, T1)
    print('2D translation')
    print(T1)

    # 2D rotation
    R = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
    T2 = np.eye(3, dtype=np.float32)
    T2[:2, :] = R
    im_2 = transform(im, T2)
    print('2D rotation')
    print(T2)

    # 2D rigid transformation: 2D rotation + 2D transformation
    T3 = np.matmul(T1, T2)
    im_3 = transform(im, T3)
    print('2D rigid transform')
    print(T3)

    # 2D affine transformation
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    T4 = np.eye(3, dtype=np.float32)
    T4[:2, :] = M
    print('Affine transform')
    print(T4)
    im_4 = transform(im, T4)

    # 2D perspective transformation
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    T5 = cv2.getPerspectiveTransform(pts1, pts2)
    print('Perspective transform')
    print(T5)
    im_5 = transform(im, T5)

    # show the images
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('original image')

    ax = fig.add_subplot(2, 3, 2)
    plt.imshow(im_1[:, :, (2, 1, 0)])
    ax.set_title('translated image')

    ax = fig.add_subplot(2, 3, 3)
    plt.imshow(im_2[:, :, (2, 1, 0)])
    ax.set_title('rotated image')

    ax = fig.add_subplot(2, 3, 4)
    plt.imshow(im_3[:, :, (2, 1, 0)])
    ax.set_title('rigid transformed image')

    ax = fig.add_subplot(2, 3, 5)
    plt.imshow(im_4[:, :, (2, 1, 0)])
    ax.set_title('affine transformed image')

    ax = fig.add_subplot(2, 3, 6)
    plt.imshow(im_5[:, :, (2, 1, 0)])
    ax.set_title('perspective transformed image')

    plt.show()
