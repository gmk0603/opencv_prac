#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2

video_file = 'C:/img/big_buck.avi'

cap = cv2.VideoCapture(video_file)
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow(video_file, img)
            cv2.waitKey(25)
        else:
            break
else:
    print("can't open video.")
cap.release()
cv2.destroyAllWindows()


# In[ ]:




