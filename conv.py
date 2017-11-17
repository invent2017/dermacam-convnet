
# coding: utf-8

# In[1]:

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
get_ipython().magic('matplotlib inline')
print(tf.__version__)
tf.logging.set_verbosity("INFO")


# In[2]:

# acne + ezcema
train = []
labels = []

for count in range(1,49): #only taking first 48 pics, last 5 to act as a '''test set'''
    string = "processed/acne/" + str(count) + ".jpg"
    pic = Image.open(string).load()
    red = []
    green = []
    blue = []
    for x in range(0,64):
        for y in range(0,64):
            red.append((pic[x,y][0]))
            green.append((pic[x,y][1]))
            blue.append((pic[x,y][2]))
    rgb = []
    rgb.append(red)
    rgb.append(green)
    rgb.append(blue)

    train.append(rgb)
    labels.append(0)
    
for count in range(1,52): #only taking first 49 pics, last 5 to act as a '''test set'''
    string = "processed/ezcema/" + str(count) + ".jpg"
    pic = Image.open(string).load()
    red = []
    green = []
    blue = []
    for x in range(0,64):
        for y in range(0,64):
            red.append((pic[x,y][0]))
            green.append((pic[x,y][1]))
            blue.append((pic[x,y][2]))
    rgb = []
    rgb.append(red)
    rgb.append(green)
    rgb.append(blue)

    train.append(rgb)
    labels.append(1)
    
train = np.asarray(train, dtype=np.float32)
labels = np.asarray(labels)


# In[3]:

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
    #print(pool2.shape)
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 96])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    #print(dense.shape)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    #print(dropout.shape)
    
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
    #print(logits.shape)
    
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

train_data = train
train_labels = labels
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=200)


# In[4]:

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=50,
    num_epochs=None,
    shuffle=True)

mnist_classifier.train(
    input_fn=train_input_fn,
    steps=500,
    hooks=[logging_hook])

# logits should be same shape as onehot_labels


# In[12]:

# acne + ezcema
test = []
test_labels = []

for count in range(49,54): #only taking first 48 pics, last 5 to act as a '''test set'''
    string = "processed/acne/" + str(count) + ".jpg"
    pic = Image.open(string).load()
    red = []
    green = []
    blue = []
    for x in range(0,64):
        for y in range(0,64):
            red.append((pic[x,y][0]))
            green.append((pic[x,y][1]))
            blue.append((pic[x,y][2]))
    rgb = []
    rgb.append(red)
    rgb.append(green)
    rgb.append(blue)

    test.append(rgb)
    test_labels.append(0)
    
for count in range(52,57): #only taking first 49 pics, last 5 to act as a '''test set'''
    string = "processed/ezcema/" + str(count) + ".jpg"
    pic = Image.open(string).load()
    red = []
    green = []
    blue = []
    for x in range(0,64):
        for y in range(0,64):
            red.append((pic[x,y][0]))
            green.append((pic[x,y][1]))
            blue.append((pic[x,y][2]))
    rgb = []
    rgb.append(red)
    rgb.append(green)
    rgb.append(blue)

    test.append(rgb)
    test_labels.append(1)
    
test = np.asarray(test, dtype=np.float32)
test_labels = np.asarray(test_labels)


# In[20]:

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test},
    y=test_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
#pdt = mnist_classifier.evaluate(input_fn = eval_input_fn)
#print(pdt)

