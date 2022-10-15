#!/usr/bin/env python
# coding: utf-8

# In[184]:


import numpy as np

a = np.arange(12).reshape(3, 4)
a


# In[185]:


np.sum(a)


# In[186]:


np.sum(a, 0)


# In[187]:


np.sum(a, 1)


# In[188]:


np.mean(a)


# In[189]:


np.mean(a, 0)


# In[190]:


np.mean(a ,1)


# In[192]:


np.amin(a)


# In[193]:


np.amin(a, 0)


# In[194]:


np.amin(a, 1)


# In[195]:


np.amax(a)


# In[196]:


np.amax(a, 0)


# In[197]:


np.amax(a, 1)


# In[198]:


np.amin is np.min


# In[200]:


np.max is np.amax

