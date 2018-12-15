# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:01:05 2018

@author: Owner
"""

import cv2
import os.path


filename = 'six.jpg'

def crop2fit(img):
    x = len(img)
    y = len(img[0])
    
    top = -1
    bottom = -1
    left = -1
    right = -1
    
    #find the top boundary
    for x in range(len(img)):
        for y in range(len(img[0])):
            if img[x][y] < 255:
                top = x
                break
        if top>-1:
            break
    
    #find the bottom boundary:
    for x in range(len(img)-1,-1,-1):
        for y in range(len(img[0])):
            if img[x][y] < 255:
                bottom = x
                break
        if bottom>-1:
            break
        
    #find the left boundary
    for y in range(len(img[0])):
        for x in range(len(img)):
            if img[x][y] < 255:
                left = y
                break
        if left>-1:
            break
        
    for y in range(len(img[0])-1,-1,-1):
        for x in range(len(img)):
            if img[x][y] < 255:
                right = y
                break
        if right > -1:
            break
    
    cropped = img[top:bottom, left:right]
    return cropped

def resize2binary(filename): 
    name = os.path.splitext(filename)
    
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    ret,img1 = cv2.threshold(img, 75,255,cv2.THRESH_BINARY )
    
    img1 = crop2fit(img1)
    
    width = 16
    height = 16
    dim = (width, height)

    resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    
    for i in range(width):
        for j in range(height):
            if resized[i][j] < 255:
                resized[i][j] = 1
                
            else:
                resized[i][j] = 0

    output = []
    with open(name[0] + '.txt', 'w') as filehandle:
        for i in range(width):
            for j in range(height): 
                output.append(resized[i][j])
                filehandle.write('%d ' % resized[i][j])

    return output
    
    #Comment out the else loop and un-comment below to see output
#    cv2.imshow("Resized image", resized)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

if __name__ == "__main__":    
    resize2binary(filename)
                


