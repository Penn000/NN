#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-12-06 10:16:36
# @Author  : Penn000
# @File    : example2.py
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # f(x) = x * w + b
    f = tf.matmul(input, weights) + biases
    if activation_function is None:
        output = f
    else:
        output = activation_function(f)
    return output

