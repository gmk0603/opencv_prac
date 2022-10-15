#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np

mylist = list(range(10))
mylist


# In[68]:


for i in range(len(mylist)):
    mylist[i] = mylist[i] + 1
    
mylist


# In[69]:


a = np.arange(10)
a


# In[70]:


a + 1


# In[71]:


a = np.arange(5)
a


# In[72]:


a + 5


# In[73]:


a - 2


# In[74]:


a * 2


# In[75]:


a / 2


# In[76]:


a ** 2


# In[77]:


b = np.arange(6).reshape(2, -1)
b


# In[78]:


b * 2


# In[79]:


a


# In[80]:


a > 2


# In[81]:


a = np.arange(10, 60, 10)
b = np.arange(1, 6)
a


# In[82]:


b


# In[83]:


a + 2


# In[84]:


a + b


# In[85]:


a - b


# In[86]:


a * b


# In[87]:


a / b


# In[88]:


a ** b


# In[90]:


a = np.zeros((2, 3))
b = np.ones((3, 2))
a + b


# In[91]:


c = np.arange(3)
c


# In[92]:


a + c


# In[93]:


d = np.arange(2).reshape(2, -1)
d


# In[94]:


a + d

