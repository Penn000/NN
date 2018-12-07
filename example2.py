#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-12-06 10:16:36
# @Author  : Penn000
# @File    : example2.py
# 
# The structure of network
#     input layer: 1 neure
#     hiden layer 1: 10 neures
#     output layer: 1 neure
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # f(x) = x * w + b
    f = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = f
    else:
        outputs = activation_function(f)
    return outputs

# create data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

x_input = tf.placeholder(tf.float32, [None, 1])
y_input = tf.placeholder(tf.float32, [None, 1])

# hiden layer with 10 neures
l1 = add_layer(x_input, 1, 10, activation_function=tf.nn.relu)

# output layer
y_prediction = add_layer(l1, 10, 1, activation_function=None)

# compute loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_input - y_prediction), reduction_indices=[1]))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# train
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1, 1001):
        sess.run(train, feed_dict={x_input: x_data, y_input: y_data})
        if step % 50 == 0:
            print(step, sess.run(loss, feed_dict={x_input: x_data, y_input: y_data}))