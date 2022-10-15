#!/usr/bin/env python
# coding: utf-8

# In[211]:


import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:/img/girl.jpg')

plt.imshow(img[:, :, ::-1])
plt.xticks([])
plt.yticks([])
plt.show()


# In[ ]:




