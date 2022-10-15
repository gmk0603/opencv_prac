#!/usr/bin/env python
# coding: utf-8

# In[164]:


import numpy as np

a = np.arange(10, 20)
a


# In[165]:


np.where(a > 15)


# In[166]:


np.where(a > 15, 1, 0)


# In[167]:


a


# In[168]:


np.where(a>15, 99, a)


# In[169]:


np.where(a>15, a, 0)


# In[170]:


a


# In[171]:


b = np.arange(12).reshape(3, 4)
b


# In[173]:


coords = np.where(b>6)
coords


# In[174]:


np.stack((coords[0], coords[1]), -1)


# In[175]:


a = np.arange(10)
b = np.arange(10)
a


# In[176]:


b


# In[177]:


a == b


# In[178]:


np.all(a == b)


# In[180]:


b[5] = -1
a


# In[181]:


b


# In[182]:


np.where(a==b)


# In[183]:


np.where(a!=b)


# In[ ]:




