#!/usr/bin/env python
# coding: utf-8

# In[10]:


from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense


# In[11]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


from keras.applications.vgg16 import VGG16


# In[13]:


model = VGG16(weights='imagenet' ,include_top=False ,input_shape =(240,240,3))


# In[ ]:


kim = load_img('person1_bacteria_2.jpeg')


# In[ ]:


kim.size


# In[ ]:


kim.size


# In[ ]:


from keras.preprocessing import image


# In[ ]:


img_array = image.img_to_array(kim)


# In[ ]:


img = np.expand_dims(img_array ,axis =0)


# In[14]:


model.summary(0)


# In[ ]:


model.add(Conv2D(64 ,kernel_size =(3,3), activation ='relu' ,input_shape =(320,320 ,3)))


# In[16]:


top = Flatten()(model.output)


# In[18]:


from keras.models import Model


# In[19]:


Nm = Model(inputs = model.input , outputs=top)


# In[21]:


Nm.summary()


# In[ ]:


from keras.optimizers import RMSprop


# In[ ]:


model.add(Dense(1024 , activation='relu'))


# In[ ]:


model.add(Dense(256 , activation='relu'))


# In[ ]:


model.add(Dense(64 , activation='relu'))


# In[ ]:


model.add(Dense(1 , activation='sigmoid'))


# In[ ]:


from keras.optimizers import RMSprop


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:


train_generator = train_datagen.flow_from_directory(
    r'chest_xray\train',
    target_size=(320, 320),
    batch_size=32,
    class_mode='binary')


# In[ ]:


validation_generator = test_datagen.flow_from_directory(
    r'chest_xray\val',
    target_size=(320, 320),
    batch_size=32,
    class_mode='binary')


# In[ ]:


test_generator = test_datagen.flow_from_directory(
    r'chest_xray\test',
    target_size=(320,320),
    batch_size=32,
    class_mode='binary')


# In[ ]:


model.fit_generator(
    train_generator,
    steps_per_epoch=5216//32 ,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=16//16)

