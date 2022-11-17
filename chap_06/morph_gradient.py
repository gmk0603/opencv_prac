#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

img = cv2.imread('c:/img/morphological.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

gradiant = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)

merged = np.hstack((img, gradiant))
cv2.imshow('gradient', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




