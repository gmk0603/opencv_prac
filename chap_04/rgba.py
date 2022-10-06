#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import numpy as np


# In[ ]:


img = cv2.imread('C:/img/opencv_logo.png')
bgr = cv2.imread('C:/img/opencv_logo.png', cv2.IMREAD_COLOR)
bgra = cv2.imread('C:/img/opencv_logo.png', cv2.IMREAD_UNCHANGED)

print("default", img.shape, "color:", bgr.shape, "unchanged:", bgra.shape)

cv2.imshow('bgr', bgr)
cv2.imshow('bgra', bgra)
cv2.imshow('alpha', bgra[:, :, 3])
cv2.waitKey(0)
cv2.destroyAllwindows()
