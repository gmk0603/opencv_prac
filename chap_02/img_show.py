#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2

img_file = 'C:/img/girl.jpg'
img = cv2.imread(img_file)

if img is not None:
    cv2.imshow('IMG', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('No image file.')


# In[ ]:




