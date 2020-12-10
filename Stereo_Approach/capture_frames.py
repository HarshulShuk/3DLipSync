#!/usr/bin/env python
# coding: utf-8

# https://stackoverflow.com/questions/22704936/reading-every-nth-frame-from-videocapture-in-opencv/36961408

# In[ ]:


import numpy as np
import os
import cv2


# In[ ]:


videoFile = "videos/Left.mp4" # After cropping left video
vidcap = cv2.VideoCapture(videoFile)
success,image = vidcap.read()
print(success)


# In[ ]:


fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
print(fps)


# In[ ]:


seconds = 1
multiplier = seconds
print(multiplier)


# In[ ]:


k = 1
while success:
    frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, image = vidcap.read()

    if frameId % multiplier == 0:
        cv2.imwrite("model_cycle/frame_left%d.jpg" % k, image)
        k = k+1

vidcap.release()
print ("Complete")


# In[ ]:


videoFile = "videos/Right.mp4" # After cropping left video
vidcap = cv2.VideoCapture(videoFile)
success,image = vidcap.read()
print(success)


# In[ ]:


fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
print(fps)


# In[ ]:


seconds = 1
multiplier = seconds
print(multiplier)


# In[ ]:


k = 1
while success:
    frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, image = vidcap.read()

    if frameId % multiplier == 0:
        cv2.imwrite("model_cycle/frame_left%d.jpg" % k, image)
        k = k+1

vidcap.release()
print ("Complete")

