#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from TensorboardUtilities import *
import sys

doDropout = False

def makeModel(x, filtSizes, channels):
    return modelMeat(x, filtSizes, channels)

def convLayer(din, filtSize, in_channels, out_channels, name=""):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.truncated_normal([filtSize, in_channels, out_channels], stddev=.1, name=name+'_weights'))
            variable_summaries(w)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.truncated_normal([out_channels]))
            variable_summaries(b)
        with tf.name_scope('convlution'):
            convOut = tf.add(tf.nn.conv1d(din, w, stride=1, padding='SAME', name=name), b)
            tf.summary.histogram('convOut', convOut)
        return convOut

def BatchNorm(din, name, training = True):
    with tf.name_scope(name):
        batchNorm = tf.layers.batch_normalization(din, beta_initializer=tf.random_normal_initializer(), gamma_initializer=tf.random_normal_initializer(), name=name, training=training)
        tf.summary.histogram(name, batchNorm)
        return batchNorm

def GlobalAverageNorm(din, name):
    with tf.name_scope(name):
        globAveNorm = tf.layers.average_pooling1d(inputs=din, pool_size=(din.shape[2],), strides=(din.shape[2],), name=name)
        tf.summary.histogram('globAveNorm'+name, globAveNorm)
        return globAveNorm

def dropoutLayer(din, keep_prob, name=''):
    with tf.name_scope(name):
        dropout = tf.nn.dropout(din, keep_prob, name='Dropout'+name)
        tf.summary.histogram('Dropout'+name, dropout)
        return dropout

def convBlock(din, filtSize, inChannel, outChannel, name=""):
    with tf.name_scope('convBlock'+name) as scope:
        conv = convLayer(din, filtSize, inChannel, outChannel, "conv"+name)
        bn = BatchNorm(conv, "batchNorm"+name)
        with tf.name_scope('activations'+name):
            activations = tf.nn.relu(bn)
            tf.summary.histogram('activations'+name, activations)
        return activations

def softmaxLayer(din, inChannels, outChannels, name=""):
    with tf.name_scope('SofmaxLayer'+name):
        w = tf.Variable(tf.truncated_normal([inChannels, outChannels], stddev=.1, name='softmax_weights'+name, dtype=tf.float32))
        variable_summaries(w)
        b = tf.Variable(tf.truncated_normal([outChannels], stddev=.1, name='softmax_biaes'+name))
        variable_summaries(b)
        result = tf.matmul(din[:,0,:], w)+b
        tf.summary.histogram('SoftmaxLayer'+name, result)
        return result

def modelMeat(x, filtSizes, channels):
    layers = [x]
    for i in range(len(filtSizes)):
        layers.append(convBlock(layers[i], filtSizes[i], channels[i], channels[i+1], str(i)))
    if doDropout:
        layers.append(dropoutLayer(layers[-1], .2, 'Dropout'))
    else:
        layers.append(GlobalAverageNorm(layers[-1], "GlobalAverageNorm"))
    return softmaxLayer(layers[-1], channels[-2], channels[-1])

'''
def ucrModel(x, y, features):
    filtSizes = [8, 5, 3]
    channels = [1, 128, 256, 128, categories]
    return modelMeat(x, filtSizes, channels)
'''

def crossEntropy(predictions, classes):
    return tf.reduce_mean(-tf.reduce_sum(classes*tf.log(predictions), [1]))
