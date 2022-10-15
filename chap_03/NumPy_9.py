#!/usr/bin/env python
# coding: utf-8

# In[123]:


import numpy as np

a = np.arange(5)
a


# In[124]:


a[[1,3]]


# In[125]:


a[[True, False, True, False, True]]


# In[126]:


a = np.arange(10)
a


# In[127]:


b = a > 5
b


# In[128]:


a[b]


# In[129]:


a[a>5]


# In[132]:


a[a>5] = 1
a


# In[133]:


b = np.arange(12).reshape(3, 4)
b


# In[134]:


b[[0, 2]]


# In[135]:


b[[0,2], [2,3]]

