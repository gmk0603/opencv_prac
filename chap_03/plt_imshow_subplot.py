#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import cv2

img1 = cv2.imread('C:/img/model.jpg')
img2 = cv2.imread('C:/img/model2.jpg')
img3 = cv2.imread('C:/img/model3.jpg')

plt.subplot(1, 3, 1)
plt.imshow(img1[:,:,(2,1,0)])
plt.xticks([]); plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(img2[:, :, ::-1])
plt.xticks([]); plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(img3[:, :, ::-1])
plt.xticks([]); plt.yticks([])

plt.show()


# In[ ]:




