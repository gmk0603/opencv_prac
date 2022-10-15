#!/usr/bin/env python
# coding: utf-8

# In[136]:


import numpy as np

a = np.arange(4).reshape(2, 2)
a


# In[138]:


b = np.arange(10, 14).reshape(2, 2)
b


# In[139]:


np.vstack((a, b))


# In[140]:


np.hstack((a, b))


# In[141]:


np.concatenate((a,b), 0)


# In[142]:


np.concatenate((a,b),1)


# In[143]:


a = np.arange(12).reshape(4, 3)
b = np.arange(10, 130, 10).reshape(4, 3)
a


# In[144]:


b


# In[145]:


c = np.stack((a,b), 0)
c.shape


# In[146]:


c


# In[147]:


d = np.stack((a,b), 1)
d.shape


# In[148]:


d


# In[149]:


e = np.stack((a,b), 2)
e.shape


# In[150]:


e


# In[152]:


ee = np.stack((a,b), -1)
ee.shape


# In[153]:


a = np.arange(12)
a


# In[154]:


np.hsplit(a, 3)


# In[155]:


np.hsplit(a, [3, 6])


# In[156]:


np.hsplit(a, [3, 6, 9])


# In[157]:


np.split(a, 3, 0)


# In[158]:


np.split(a, [3, 6, 9], 0)


# In[159]:


b = np.arange(12).reshape(4, 3)
b


# In[160]:


np.vsplit(b, 2)


# In[161]:


np.split(b, 2, 0)


# In[162]:


np.hsplit(b, [1])


# In[163]:


np.split(b, [1], 1)


# In[ ]:




