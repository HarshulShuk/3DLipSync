#!/usr/bin/env python
# coding: utf-8

# In[4]:


from moviepy.editor import VideoFileClip
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import urllib
import glob
import string


# In[5]:


all_images_path = glob.glob('myframes/every_n_sec/*.jpg')
print(all_images_path[1:5])


# In[6]:


all_images_path = glob.glob('myframes/every_n_sec/*.jpg')
for image_path in all_images_path:
    img = cv2.imread(image_path)
    img_object = Image.open(image_path)
    h, w = img.shape[0], img.shape[1]
    cropped_img_left = img_object.crop((0,0,w/2,h))
    cropped_img_right = img_object.crop((w/2,0,w,h))
    cropped_img_left.save('data/left/' + 'left_' + image_path.strip('myframes/every_n_sec/'))
    cropped_img_right.save('data/right/' + 'right_' + image_path.strip('myframes/every_n_sec/'))
print("Done")


# In[8]:


img = cv2.imread('myframes/every_n_sec/frame1115.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
f = plt.figure()
f.set_figheight(6)
f.set_figwidth(12)

image_path = "myframes/every_n_sec/frame1215.jpg"
h, w = img.shape[0],img.shape[1]
print(h, w)
print(h, w/2)
image_obj = Image.open(image_path)
cropped_image = image_obj.crop((0,0,w/2,h))

f.add_subplot(1,2, 1)
plt.imshow(cropped_image)
plt.axis('off')

h, w = img.shape[0],img.shape[1]
image_obj = Image.open(image_path)
cropped_image_r = image_obj.crop((w/2,0,w,h))

f.add_subplot(1,2, 2)
plt.imshow(cropped_image_r)

plt.axis('off')
#plt.rcParams["figure.figsize"] = (800,960)
plt.show()


# In[ ]:




