
# coding: utf-8

# In[25]:

import tensorflow as tf
import pandas as pd
import numpy as np
import tflearn
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
get_ipython().magic('matplotlib inline')
print(tf.__version__)
tf.logging.set_verbosity("INFO")


# In[46]:

# define preprocessing function
def preprocess(disease, first, last, data, label):
    for count in range(first,last+1): #only taking first 48 pics, last 5 to act as a '''test set'''
        string = "processed/" + str(disease) + "/" + str(count) + ".jpg"
        pic = Image.open(string).load()
        rgb = []
        for x in range(0,64):
            for y in range(0,64):
                rgb.append(pic[x,y])
        
        #rgb.append(red)
        #rgb.append(green)
        #rgb.append(blue)

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


# In[1]:

# define network architecture
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[13, 13],
      padding="same",
      activation=tf.nn.relu) #output: [-1, 64, 64, 64]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2) #output [-1, 32, 32, 64]
    
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=96,
      kernel_size=[13, 13],
      padding="same",
      activation=tf.nn.relu) #output: [-1, 32, 32, 96]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2) #output [-1, 16, 16, 96]
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 96])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
    
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Loss Calculation
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    # Configure training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Add Evaluation Metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# create dermacam model
dermacam = tf.estimator.Estimator(model_fn=cnn_model_fn)

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)


# In[42]:

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=10,
    num_epochs=None,
    shuffle=True)

dermacam.train(
    input_fn=train_input_fn,
    steps=500,
    hooks=[logging_hook])


# In[43]:

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False)

eval_results = dermacam.evaluate(input_fn=eval_input_fn,hooks=[logging_hook])
print(eval_results)

# 0 1 1 0 0 0 1 1 0 1 (Predicted Results)
# 0 0 0 0 0 1 1 1 1 1 (Actual results)
# 6/10, 60% accuracy.

