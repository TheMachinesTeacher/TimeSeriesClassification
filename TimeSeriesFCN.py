#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pandas as pd
from DataLoader import *
from TensorboardUtilities import *
import sys

doDropout = True
passesOverData = 150

class varHolder():
    trainData = None
    trainLabels = None
    trainBatchNum = 0
    testData = None
    testLabels = None
    testBatchNum = 0
    sequenceLength = 0
    features = 0
    batchSize = 32
    categories = 0

vh = varHolder()

def loadWatchData():
    vh.sequenceLength = 1024
    vh.features = 3
    vh.categories = 5
    datapath = 'datas/watch_accelerometer_with_5_classes.csv'
    data, labels = getData(datapath)
    dataDict = convertToDictionary(data, labels)
    trainDict, testDict = splitData(dataDict)
    trainData, trainLabels = convertToLists(trainDict, vh.sequenceLength)
    testData, testLabels = convertToLists(testDict, vh.sequenceLength)
    return trainData, trainLabels, testData, testLabels

def getBatch(training):
    xd = []
    yd = []
    for i in range(vh.batchSize):
        if training:
            xd.append(vh.trainData[vh.trainBatchNum])
            yd.append(vh.trainLabels[vh.trainBatchNum])
            vh.trainBatchNum += 1
        else:
            xd.append(vh.testData[vh.testBatchNum])
            yd.append(vh.testLabels[vh.testBatchNum])
            vh.testBatchNum += 1
    xd = np.array(xd, dtype=np.float32)
    yd = np.array(yd, dtype=np.float32)
    return {x:xd, y:yd}

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

def makeModel(x, features):
    filtSizes = [8, 5, 3]
    channels = [vh.features, 128, 256, 128, vh.categories]
    return modelMeat(x, filtSizes, channels)

def ucrModel(x, y, features):
    filtSizes = [8, 5, 3]
    channels = [1, 128, 256, 128, vh.categories]
    return modelMeat(x, filtSizes, channels)

def crossEntropy(predictions, classes):
    return tf.reduce_mean(-tf.reduce_sum(classes*tf.log(predictions), [1]))

if __name__ == "__main__":
    vh.trainData, vh.trainLabels, vh.testData, vh.testLabels = loadWatchData()
    with tf.Session() as sess:
        x = tf.placeholder(dtype=tf.float32, shape=(vh.batchSize, vh.sequenceLength, vh.features))
        y = tf.placeholder(dtype=tf.float32, shape=(None, vh.categories))
        categoryProbabilities = makeModel(x, vh.features)
        with tf.name_scope('cross_entropy'):
#            cross_entropy = crossEntropy(categoryProbabilities, y)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=categoryProbabilities))
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                #print(categoryProbabilities)
                a = tf.argmax(categoryProbabilities, 1)
                b = tf.argmax(y,1)
                correct_predictions = tf.equal(a, b)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('train', sess.graph)
        test_writer = tf.summary.FileWriter('test')
        sess.run(tf.global_variables_initializer())
        i = 0
        for j in range(passesOverData):
            try:
                while True:
                    batch = getBatch(True)
                    if i%10 == 0:
                        #print(a.eval(feed_dict=batch))
                        #print(b.eval(feed_dict=batch))
                #        print(y.eval(feed_dict=batch))
                #        print(categoryProbabilities.eval(feed_dict=batch))
                   #     print(correct_predictions.eval(feed_dict=batch))
                        train_accuracy = accuracy.eval(feed_dict=batch)
                        print('Accuracy at step %s: %s' % (i, train_accuracy*100))
                    train_step.run(feed_dict=batch)
                    i += 1
            except IndexError as msg:
                vh.trainBatchNum = 0
                print('Finished pass %s over data' % (j+1,))
        accr = 0.0
        iters = 0
        try:
            while True:
                accr += accuracy.eval(feed_dict=getBatch(False))
                iters += 1
        except IndexError:
            print('Test accuracy: %s ' % ((accr/iters)*100,))

