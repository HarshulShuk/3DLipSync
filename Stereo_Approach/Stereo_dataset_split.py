#!/usr/bin/env python
# coding: utf-8

# In[1]:


file = open("stereo_images_train.txt", "w") 
for i in range(0, 5000):
    file.write("frame_left"+str(i)+".jpg"+str(" ")+"frame_right"+str(i)+".jpg \n") 
file.close() 


# In[ ]:


file = open("stereo_images_eval.txt", "w") 
for i in range(50001, 6000):
    file.write("frame_left"+str(i)+".jpg"+str(" ")+"frame_right"+str(i)+".jpg \n") 
file.close() 


# In[ ]:


file = open("stereo_images_test.txt", "w") 
for i in range(6001, 7000):
    file.write("frame_left"+str(i)+".jpg"+str(" ")+"frame_right"+str(i)+".jpg \n") 
file.close() 

