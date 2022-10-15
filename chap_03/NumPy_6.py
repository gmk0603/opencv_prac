#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np

a = np.arange(6)
a


# In[54]:


b = a.reshape(2, 3)
b


# In[55]:


c = np.reshape(a, (2, 3))
c


# In[56]:


d = np.arange(100).reshape(2, -1)
d


# In[57]:


d.shape


# In[58]:


e = np.arange(100).reshape(-1, 5)
e


# In[59]:


e.shape


# In[60]:


f = np.zeros((2, 3))
f


# In[61]:


f.reshape((6, ))


# In[62]:


f.reshape(-1)


# In[63]:


np.ravel(f)


# In[64]:


g = np.arange(10).reshape(2, -1)
g


# In[65]:


g.T


# In[ ]:




