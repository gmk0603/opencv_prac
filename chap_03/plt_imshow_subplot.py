#!/usr/bin/env python
# coding: utf-8

# In[87]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[88]:


img1 = cv2.imread('C:/img/model.jpg')
img2 = cv2.imread('C:/img/model2.jpg')
img3 = cv2.imread('C:/img/model3.jpg')

plt.subplot(1, 3, 1)
plt.imshow(img1[:, :, ::-1])
plt.xticks([]); plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(img2[:, :, (2, 1, 0)])
plt.xticks([]); plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(img3[:, :, ::-1])
plt.xticks([]); plt.yticks([])

plt.show()


# In[ ]:




