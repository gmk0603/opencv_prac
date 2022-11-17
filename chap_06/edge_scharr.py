#!/usr/bin/env python
# coding: utf-8

# In[46]:


import cv2
import numpy as np

img = cv2.imread('c:/img/sudoku.jpg')

gx_k = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
gy_k = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

scharrx = cv2.Scharr(img, -1, 1, 0)
scharry = cv2.Scharr(img, -1, 0, 1)

merged1 = np.hstack((img, edge_gx, edge_gy))
merged2 = np.hstack((img, scharrx, scharry))
merge = np.vstack((merged1, merged2))
cv2.imshow('Scharr',merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

