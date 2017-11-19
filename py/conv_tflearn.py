
# coding: utf-8

# In[1]:

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

get_ipython().magic('matplotlib inline')
print(tf.__version__)
sess = tf.InteractiveSession()
tf.logging.set_verbosity("INFO")


# In[2]:

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
        elif disease == "psoriasis":
            label.append(2)
        elif disease == "seborrheic keratoses":
            label.append(3)
        elif disease == "skin cancer":
            label.append(4)

def onehot(array,depth):
    n = array.size
    onehot = np.zeros((n,depth))
    for count in range(0,array.size):
        onehot[count][array[count]] = 1
        
    return onehot



# define training data
train_data = []
train_labels = []

preprocess("acne",1,48,train_data,train_labels)
preprocess("ezcema",1,51,train_data,train_labels)
preprocess("psoriasis",1,44,train_data,train_labels)
preprocess("seborrheic keratoses",1,45,train_data,train_labels)
preprocess("skin cancer",1,45,train_data,train_labels)
    
# define testing data
test_data = []
test_labels = []

preprocess("acne",49,53,test_data,test_labels)
preprocess("ezcema",52,56,test_data,test_labels)
preprocess("psoriasis",45,49,test_data,test_labels)
preprocess("seborrheic keratoses",46,50,test_data,test_labels)
preprocess("skin cancer",46,50,test_data,test_labels)

# reshape data
train_labels = np.asarray(train_labels)
train_labels = onehot(train_labels,5)
test_labels = np.asarray(test_labels)
test_labels = onehot(test_labels,5)

train_data = np.asarray(train_data, dtype=np.float32)/255
train_data = train_data.reshape([-1, 64, 64, 3])
test_data = np.asarray(test_data, dtype=np.float32)/255
test_data = test_data.reshape([-1, 64, 64, 3])


# In[3]:

test_labels.shape


# In[ ]:

# define network architecture
# input
network = input_data(shape=[None, 64, 64, 3], name='input')

# conv 1 and pooling 1
network = conv_2d(network, 128, 16, activation = "relu6", regularizer="L2", weights_init="xavier")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

# conv 2 and pooling 2
network = conv_2d(network, 96, 12, activation = "relu", regularizer="L2", weights_init="xavier")
network = avg_pool_2d(network, 2)
network = local_response_normalization(network)

# conv 2 and pooling 2
network = conv_2d(network, 64, 8, activation = "relu", regularizer="L2", weights_init="xavier")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

# dense and dropout
network = fully_connected(network, 512, activation = "relu", weights_init="xavier")
network = dropout(network, 0.9)

# logits
network = fully_connected(network, 5, activation = "softmax")

# target
network = regression(network, optimizer='adagrad', learning_rate=0.0015,
                     loss='softmax_categorical_crossentropy', name='target')

# training
dermacam = tflearn.DNN(network, tensorboard_verbose=3)


# In[ ]:

dermacam.fit({'input': train_data}, {'target': train_labels}, n_epoch=15,
           validation_set=({'input': test_data}, {'target': test_labels}),
           snapshot_step=10, show_metric=True, run_id='convnet_dermacam')


# In[ ]:

dermacam.predict({'input': test_data})
#0 1 1 1 1 0 1 1 1 1


# In[ ]:

dermacam.predict_label({'input': test_data})
# 0 0 0 0 0 1 1 1 1 1

