#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

win_name = 'Tracbar'

img = cv2.imread('C:/img/blank_500.jpg')
cv2.imshow(win_name, img)

def onChange(x):
    print(x)
    
    r = cv2.getTrackbarPos('R', win_name)
    g = cv2.getTrackbarPos('G', win_name)
    b = cv2.getTrackbarPos('B', win_name)
    print(r, g, b)
    img[:] = [b,g,r]
    cv2.imshow(win_name, img)
    
cv2.createTrackbar('R', win_name, 255, 255, onChange)
cv2.createTrackbar('G', win_name, 255, 255, onChange)
cv2.createTrackbar('B', win_name, 255, 255, onChange)

while True:
    if cv2.waitKey(0) & 0xFF == 27:
        break
destroyAllWindows()


# In[ ]:




