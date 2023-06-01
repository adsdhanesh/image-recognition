#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


dataset=tf.keras.datasets.fashion_mnist


# In[6]:


(train_images,train_labels),(test_images,test_labels)=dataset.load_data()


# In[9]:


len(train_images)


# In[10]:


len(train_labels)


# In[30]:


model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])


# In[32]:


model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# In[33]:


model.fit(train_images,train_labels,epochs=10)


# In[37]:


prediction_model=tf.keras.Sequential([model,tf.keras.layers.Softmax()])


# In[49]:


p=prediction_model.predict(test_images)


# In[50]:


p[18]


# In[51]:


np.argmax(p[18])


# In[52]:


test_images[18]


# In[ ]:




