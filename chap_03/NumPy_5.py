#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np

a = np.arange(5)
a


# In[46]:


a.dtype


# In[47]:


b = a.astype('float32')
b


# In[48]:


a.dtype


# In[49]:


c = a.astype(np.float64)
c


# In[50]:


c.dtype


# In[51]:


a.dtype


# In[52]:


d = np.uint8(a)
d

