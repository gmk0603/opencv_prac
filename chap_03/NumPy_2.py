#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
a = np.empty((2, 3))
a


# In[11]:


a.dtype


# In[12]:


a.fill(255)
a


# In[13]:


b = np.zeros((2, 3))
b


# In[14]:


b.dtype


# In[15]:


c = np.zeros((2, 3), dtype = np.int8)
c


# In[16]:


d = np.ones((2, 3), dtype = np.int16)
d


# In[17]:


e = np.full((2, 3, 4,), 255, dtype = np.uint8)
e


# In[ ]:




