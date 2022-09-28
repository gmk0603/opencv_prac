#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


img = cv2.imread('C:/img/sunset.jpg')

x = 320; y = 150; w = 50; h = 50
roi = img[y:y+h, x:x+w] #지정된 위치 복사한 이미지
img2 = roi.copy()

img[y:y+h, x+w:x+w+w] = roi #복사한 이미지 roi를 지정한 위치에 붙여넣기
cv2.rectangle(img, (x, y), (x+w+w, y+h), (0, 255, 0))

cv2.imshow("img", img)
cv2.imshow("roi", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

