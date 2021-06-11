#   Mustafa Sağlam  #
#   150140129       #
#   Computer Vision #HW4
#   22.12.2018      #

import cv2
import numpy as np
import matplotlib as plt
import sys

def binary_mask(img):
    return np.uint8((img>75)*255)

def morph_mask(mask):
    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(mask, kernel, iterations = 3)
    opened = cv2.dilate(eroded, kernel, iterations = 3)
    return opened

def k2means(img, mask):
    H,W = img.shape
    img_C = img.copy()
    
    for h in range(H):
        for w in range(W):
            if(mask[h,w]>0):
                if(img[h,w]>205):
                    img_C[h,w] = 255
                else:
                    img_C[h,w] = 95
            else:
                img_C[h,w] = 0

    return img_C

def prewitt_edge(img):
    Ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    Kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    H,W = img.shape
    img_C = img.copy()
    points = []
    for h in range(H):
        for w in range(W):
            if(img[h,w]>0):
                hor = Kx[0,0]*img[h-1,w-1] + Kx[0,1]*img[h-1,w] + Kx[0,2]*img[h-1,w+1]\
                    + Kx[1,0]*img[h , w-1] + Kx[1,1]*img[h , w] + Kx[1,2]*img[h , w+1]\
                    + Kx[2,0]*img[h+1,w-1] + Kx[2,1]*img[h+1,w] + Kx[2,2]*img[h+1,w+1]
                ver = Ky[0,0]*img[h-1,w-1] + Ky[0,1]*img[h-1,w] + Ky[0,2]*img[h-1,w+1]\
                    + Ky[1,0]*img[h , w-1] + Ky[1,1]*img[h , w] + Ky[1,2]*img[h , w+1]\
                    + Ky[2,0]*img[h+1,w-1] + Ky[2,1]*img[h+1,w] + Ky[2,2]*img[h+1,w+1]
                if(abs(hor)>300 or abs(ver)>300):
                    points.append((w,h))
    return points


if __name__ == "__main__":
    img_mr = cv2.imread("mr.jpg",0)
    cv2.imshow("Original Image", img_mr)

    mask_init = binary_mask(img_mr)
    cv2.imshow("Binary Mask", mask_init)

    mask_brain = morph_mask(mask_init)
    cv2.imshow("Brain-Mask", mask_brain)

    img_k2mr = k2means(img_mr, mask_brain)
    cv2.imshow("Kmeans-Results", img_k2mr)

    points = prewitt_edge(img_k2mr)
    color_mr = cv2.imread("mr.jpg",1)
    h,w = img_mr.shape
    pcolor = (255, 55, 55)#BGR
    rect = (0,0,w,h)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    for p in points:
        cv2.circle(color_mr, p, 1, pcolor, -1, cv2.LINE_AA, 0)

    cv2.imshow("Final Result", color_mr)

    cv2.waitKey(0)