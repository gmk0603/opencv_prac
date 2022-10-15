#!/usr/bin/env python
# coding: utf-8

# In[18]:


import cv2
import numpy as np

img = cv2.imread('C:/img/girl.jpg')
img


# In[19]:


img.shape


# In[26]:


a = np.empty_like(img)
b = np.zeros_like(img)
c = np.ones_like(img)
d = np.full_like(img, 255)
a


# In[27]:


a.shape


# In[28]:


b


# In[29]:


b.shape


# In[30]:


c


# In[31]:


c.shape


# In[32]:


d


# In[33]:


d.shape


# In[ ]:




