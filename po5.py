#print absolute value of an integer:

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
# import cv2.cv as cv2
# from numpy import *

# e1=cv2.getTickCount()

# img1=cv2.imread('s1.jpg')
# img2=cv2.imread('s2.jpg')

# rows,cols,channels=img2.shape
# roi=img1[0:rows,0:cols]

# img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret,mask=cv2.threshold(img2gray,175,255,cv2.THRESH_BINARY)
# mask_inv=cv2.bitwise_not(mask)

# img1_bg=cv2.bitwise_and(roi,roi,mask=mask)
# img2_fg=cv2.bitwise_and(img2,img2,mask=mask_inv)
# dst=cv2.add(img1_bg,img2_fg)
# img1[0:rows,0:cols]=dst
# # print img1.shape
# # cv2.imshow('dst',dst)
# # cv2.NamedWindow("Source")  
# # cv2.ShowImage("Source",im)  
# cv2.imshow('3',img1)

# e2=cv2.getTickCount()




# cap=cv2.VideoCapture(1)
# print (e2-e1)/cv2.getTickFrequency()
# cv2.waitKey(0)  
# cv2.destroyAllWindows()
# while (1):
# 	ret,frame=cap.read()

# 	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
# 	lower_blue=np.array([110,50,50])
# 	upper_blue=np.array([130,255,255])

# 	mask=cv2.inRange(hsv,lower_blue,upper_blue)
# 	res=cv2.bitwise_and(frame,frame,mask=mask)

# 	cv2.imshow('frame',frame)
# 	cv2.imshow('mask',mask)
# 	cv2.imshow('res',res)

# 	if cv2.waitKey(5)==ord('q'):
# 		break


# img3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# ret1,th1=cv2.threshold(img3,127,255,cv2.THRESH_BINARY)

# ret2,th2=cv2.threshold(img3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# blur=cv2.GaussianBlur(img3,(5,5),0)

# ret3,th3=cv2.threshold(img3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	
# images=[img3,0,th1,img3,0,th2,blur,0,th3]
# titles=['11','12','13','21','22','23','31','32','33']


# kernel=np.ones((5,5),np.float32)/25
# dst=cv2.filter2D(img1,-1,kernel)
# dst=cv2.blur(img1,(5,5))
# dst=cv2.GaussianBlur(img1,(5,5),0)
# dst=cv2.medianBlur(img1,5)
# dst=cv2.bilateralFilter(img1,9,75,75)
img1=cv2.imread('s4.jpg')
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

kernel=np.ones((5,5),np.uint8)
# erosion=cv2.erode(img1,kernel,iterations=1)
# dilation=cv2.dilate(img1,kernel,iterations=1)
# opening=cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernel)
# closing=cv2.morphologyEx(img1,cv2.MORPH_CLOSE,kernel)
# gradient=cv2.morphologyEx(img1,cv2.MORPH_GRADIENT,kernel)
tophat=cv2.morphologyEx(img1,cv2.MORPH_TOPHAT,kernel)
blackhat=cv2.morphologyEx(img1,cv2.MORPH_BLACKHAT,kernel)

cv2.imshow('1',img1)
cv2.imshow('2',tophat)
cv2.imshow('3',blackhat)
# plt.subplot(121),plt.imshow(img1),plt.title('Original')
# plt.subplot(122),plt.imshow(erosion),plt.title('Averaging')
# plt.show()
print cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
print cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
print cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
cv2.waitKey(-1)