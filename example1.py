#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-12-05 10:41:36
# @Author  : Penn000
# @File    : example1.py
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1 * x_data + 0.3

# construct model
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = w * x_data + b

# compute loss
loss = tf.reduce_mean(tf.square(y - y_data))

# propagate error
learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# train
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(w), sess.run(b))