
# coding: utf-8

# In[4]:

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

get_ipython().magic('matplotlib inline')
print(tf.__version__)
tf.logging.set_verbosity("INFO")


# In[16]:

# define preprocessing function
def preprocess(disease, first, last, data, label):
    for count in range(first,last+1): #only taking first 48 pics, last 5 to act as a '''test set'''
        string = "processed/" + str(disease) + "/" + str(count) + ".jpg"
        pic = Image.open(string).load()
        rgb = []
        for x in range(0,64):
            for y in range(0,64):
                rgb.append(pic[x,y])

        data.append(rgb)
        if disease == "acne":
            label.append(0)
        elif disease == "ezcema":
            label.append(1)

# define training data
train_data = []
train_labels = []

preprocess("acne",1,48,train_data,train_labels)
preprocess("ezcema",1,51,train_data,train_labels)
    
train_data = np.asarray(train_data, dtype=np.float32)
train_labels = np.asarray(train_labels)

# define testing data
test_data = []
test_labels = []

preprocess("acne",49,53,test_data,test_labels)
preprocess("ezcema",52,56,test_data,test_labels)

test_data = np.asarray(test_data, dtype=np.float32)
test_labels = np.asarray(test_labels)


# In[19]:

#train_labels = tf.one_hot(indices=tf.cast(train_labels, tf.int32), depth=2)
#test_labels = tf.one_hot(indices=tf.cast(test_labels, tf.int32), depth=2)
train_data = train_data.reshape([-1, 64, 64, 3])
test_data = test_data.reshape([-1, 64, 64, 3])


# In[22]:

# define network architecture
# input
network = input_data(shape=[None, 64, 64, 3], name='input')

# conv 1 and pooling 1
network = conv_2d(network, 64, 13, activation = "relu")
network = max_pool_2d(network, [2,2], 2)

# conv 2 and pooling 2
network = conv_2d(network, 96, 13, activation = "relu")
network = max_pool_2d(network, [2,2], 2)

# dense and dropout
network = fully_connected(network, 1024, activation = "relu")
network = dropout(network, 0.9)

# logits
network = fully_connected(network, 2, activation = "softmax")

# target
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')


# In[23]:

# training
dermacam = tflearn.DNN(network, tensorboard_verbose=0)
dermacam.fit({'input': train_data}, {'target': train_labels}, n_epoch=10,
           validation_set=({'input': test_data}, {'target': test_labels}),
           snapshot_step=100, show_metric=True, run_id='convnet_dermacam')

# valueerror when running. something about shapes not matching. i tried commenting out the block above
# for one_hot with labels. if i leave the line wiht one_hot for labels in, another error arises.
# for some reason training samples and validation samples logged always seem to be more than the actual
# number of samples? sometimes i get 297/30 and other times i get 198/20 when there should only be 99/10.
# 99/10 is expected since data augmentation is not yet done

