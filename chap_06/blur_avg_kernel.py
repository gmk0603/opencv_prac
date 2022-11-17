#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np

img = cv2.imread('c:/img/girl.jpg')

kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                  [0.04, 0.04, 0.04, 0.04, 0.04],
                  [0.04, 0.04, 0.04, 0.04, 0.04],
                  [0.04, 0.04, 0.04, 0.04, 0.04],
                  [0.04, 0.04, 0.04, 0.04, 0.04]])

kernel = np.ones((5, 5))/5**2

blured = cv2.filter2D(img, -1, kernel)

cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured)
cv2.waitKey()
cv2.destroyAllWindows()

