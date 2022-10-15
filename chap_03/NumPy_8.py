#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np

a = np.arange(10)
a


# In[96]:


a[5]


# In[97]:


b = np.arange(12).reshape(3, 4)
b


# In[98]:


b[1]


# In[99]:


b[1, 2]


# In[100]:


a


# In[110]:


a[5] = 9
a


# In[104]:


b


# In[109]:


b[0] = 0
b


# In[108]:


b[1, 2] = 99
b


# In[111]:


a = np.arange(10)
a


# In[112]:


a[2:5]


# In[113]:


a[5:]


# In[114]:


a[:]


# In[115]:


b = np.arange(12).reshape(3, 4)
b


# In[116]:


b[0:2, 1]


# In[117]:


b[0:2, 1:3]


# In[118]:


b[2, :]


# In[119]:


b[:, 1]


# In[120]:


b[0:2, 1:3] = 0
b


# In[121]:


bb = b[0:2, 1:3]
bb


# In[122]:


bb[0] = 99
b

