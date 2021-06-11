#   Mustafa Sağlam  #
#   150140129       #
#   Computer Vision #HW4
#   22.12.2018      #

import cv2
import numpy as np
import matplotlib as plt
import sys

def gaussian_filter(img):
    H,W = img.shape
    #5x5 gaussian Kernel with sigma = 1.0
    K = (1.0/273)*(np.asarray([[1, 4, 7, 4,1],
                               [4,16,26,16,4],
                               [7,26,41,26,7],
                               [4,16,26,16,4],
                               [1, 4, 7, 4,1]]))
    img_C = img.copy()
    for w in range(2,W-2):
        for h in range(2,H-2):
            img_C[w,h] = K[0,0]*img[w-2,h-2] + K[0,1]*img[w-2,h-1] + K[0,2]*img[w-2,h] + K[0,3]*img[w-2,h+1] + K[0,4]*img[w-2,h+2]\
                        +K[1,0]*img[w-1,h-2] + K[1,1]*img[w-1,h-1] + K[1,2]*img[w-1,h] + K[1,3]*img[w-1,h+1] + K[1,4]*img[w-1,h+2]\
                        +K[2,0]*img[w , h-2] + K[2,1]*img[w , h-1] + K[2,2]*img[w , h] + K[2,3]*img[w , h+1] + K[2,4]*img[w , h+2]\
                        +K[3,0]*img[w+1,h-2] + K[3,1]*img[w+1,h-1] + K[3,2]*img[w+1,h] + K[3,3]*img[w+1,h+1] + K[3,4]*img[w+1,h+2]\
                        +K[4,0]*img[w+2,h-2] + K[4,1]*img[w+2,h-1] + K[4,2]*img[w+2,h] + K[4,3]*img[w+2,h+1] + K[4,4]*img[w+2,h+2]
    return img_C

def gradientXY(img):
    H,W = img.shape
    #kernel for x differences
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    #kernel for y differences
    ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
    Ix = np.zeros((H,W), dtype=np.int64)
    Iy = np.zeros((H,W), dtype=np.int64)
    for w in range(1,W-1):
        for h in range(1,H-1):
            Ix[w,h] = kx[0,0]*img[w-1,h-1] + kx[0,1]*img[w-1,h] + kx[0,2]*img[w-1,h+1]\
                    + kx[1,0]*img[w , h-1] + kx[1,1]*img[w , h] + kx[1,2]*img[w , h+1]\
                    + kx[2,0]*img[w+1,h-1] + kx[2,1]*img[w+1,h] + kx[2,2]*img[w+1,h+1]
            Iy[w,h] = ky[0,0]*img[w-1,h-1] + ky[0,1]*img[w-1,h] + ky[0,2]*img[w-1,h+1]\
                    + ky[1,0]*img[w , h-1] + ky[1,1]*img[w , h] + ky[1,2]*img[w , h+1]\
                    + ky[2,0]*img[w+1,h-1] + ky[2,1]*img[w+1,h] + ky[2,2]*img[w+1,h+1]
    return (Ix,Iy)
            
def harris_detector(Ix, Iy, img):
    h,w = img.shape
    points = []
    Ixx = np.zeros((h,w),np.int64)
    Iyy = np.zeros((h,w),np.int64)
    Ixy = np.zeros((h,w),np.int64)
    for y in range(h):
        for x in range(w):
            Ixx[y,x] = Ix[y,x]**2
            Iyy[y,x] = Iy[y,x]**2
            Ixy[y,x] = Ix[y,x]*Iy[y,x]

    threshold = (2**31)*(2**0.7) #~3.49e9
    k = 0.041 #41 kere maaşallah :]
    #for 3x3 windows
    #using R = det(H) - k(trace(H)^2)
    for y in range(1,h-1):
        for x in range(1,w-1):
            Sxx = Syy = Sxy = 0
            i = -1
            while (i<2):
                j = -1
                while (j<2):
                    Sxx += Ixx[y+j, x+i]
                    Syy += Iyy[y+j, x+i]
                    Sxy += Ixy[y+j, x+i]
                    j+=1
                i+=1
            H = np.array([[Sxx, Sxy], [Sxy, Syy]])
            det = np.linalg.det(H)
            trace = np.trace(H)
            R = det - k*(trace*trace)
            if(R>threshold):
                points.append((x,y))
    return points


if __name__ == "__main__":
    gray_blocks = cv2.imread("blocks.jpg",0)
    color_blocks = cv2.imread("blocks.jpg",1)
    cv2.imshow("Original Image",gray_blocks)
    
    filtered = gaussian_filter(gray_blocks)
    cv2.imshow("Gaussian Filtered",filtered)
    
    Ix, Iy = gradientXY(gray_blocks)

    points = harris_detector(Ix, Iy, filtered)
    h,w = gray_blocks.shape
    pcolor = (25, 55, 245)#BGR
    rect = (0,0,w,h)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    for p in points:
        cv2.circle(color_blocks, p, 1, pcolor, -1, cv2.LINE_AA, 0)

    cv2.imshow("Harris-Corner Detector Result", color_blocks)

    cv2.waitKey(0)