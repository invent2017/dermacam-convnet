
# coding: utf-8

# In[ ]:

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
import tflearn.data_preprocessing
import tflearn.data_augmentation

get_ipython().magic('matplotlib inline')
print(tf.__version__)
sess = tf.InteractiveSession()
tf.logging.set_verbosity("INFO")

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


# In[ ]:

# define network architecture (old)
# input
_input = input_data(shape=[None, 64, 64, 3], name='input')

# conv 1 and pooling 1
_conv1 = conv_2d(_input, 128, 12, activation = "relu", regularizer="L2", weights_init="xavier", name="c1")
_pool1 = max_pool_2d(_conv1, 2)
_norm1 = local_response_normalization(_pool1)

# conv 2 and pooling 200
_conv2 = conv_2d(_norm1, 128, 12, activation = "relu", regularizer="L2", weights_init="xavier")
_pool2 = max_pool_2d(_conv2, 2)
_norm2 = local_response_normalization(_pool2)

# dense and dropout
_dense = fully_connected(_norm2, 512, activation = "relu", weights_init="xavier")
_dropout = dropout(_dense, 0.9)

# logits
_logits = fully_connected(_dropout, 5, activation = "softmax")

# target
network = regression(_logits, optimizer='rmsprop', learning_rate=0.01,#0015,
                     loss='binary_crossentropy', name='target',)

# training
dermacam = tflearn.DNN(network, tensorboard_verbose=3)
#dermacam.load("models/dermacam1.tflearn")


# In[ ]:

dermacam.predict({'input': test_data})


# In[ ]:

dermacam.predict_label({'input': test_data})


# In[ ]:

dermacam.save("models/dermacam2.tflearn")
dermacam.get_weights(_logits.W)

