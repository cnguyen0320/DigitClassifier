# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:01:05 2018

@author: Owner
"""

#import matplotlib.pyplot as plt

import cv2
import os.path


filename = 'zero.jpg'

def Resize_binary(filename): 
    name = os.path.splitext(filename)
    
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    ret,img1 = cv2.threshold(img, 75,255,cv2.THRESH_BINARY )
        
    width = 256
    height = 256
    dim = (width, height)
    
    resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    
    for i in range(256):
        for j in range(256):    
            if resized[i][j] < 255:
                resized[i][j] = 0
            else:
                resized[i][j] = 1
    
    with open(name[0] + '.txt', 'w') as filehandle:
        for i in range(256):
            for j in range(256): 
                filehandle.write('%d ' % resized[i][j])
                

#cv2.imshow("Resized image", resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
