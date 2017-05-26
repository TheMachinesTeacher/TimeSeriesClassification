#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from DataLoader import *
from TensorboardUtilities import *
from Model import *
from multiprocessing import Process


class ModelRun(Process):

    dataset = ''
    epochs = 0
    categories = 0
    sequenceLength = 0
    features = 0
    batchSize = 0
    trainData = None
    trainLabels = None
    trainBatchNum = 0
    testData = None
    testLabels = None
    testBatchNum = 0
    filtSizes = None
    channels = None

    def __init__(self, dataset, filtSizes, channels, epochs, categories, sequenceLength, features, batchSize):
        super(ModelRun, self).__init__()
        self.dataset = dataset
        self.filtSizes = filtSizes
        self.channels = channels
        self.epochs = epochs
        self.categories = categories
        self.sequenceLength = sequenceLength
        self.features = features
        self.batchSize = batchSize
    
    def loadWatchData(self):
        datapath = 'datas/watch_accelerometer_with_5_classes.csv'
        data, labels = getData(datapath)
        dataDict = convertToDictionary(data, labels)
        trainDict, testDict = splitData(dataDict)
        trainData, trainLabels = convertToLists(trainDict, self.sequenceLength)
        testData, testLabels = convertToLists(testDict, self.sequenceLength)
        return trainData, trainLabels, testData, testLabels
    
    def getBatch(self, x, y, training):
        xd = []
        yd = []
        for i in range(self.batchSize):
            if training:
                xd.append(self.trainData[self.trainBatchNum])
                yd.append(self.trainLabels[self.trainBatchNum])
                self.trainBatchNum += 1
            else:
                xd.append(self.testData[self.testBatchNum])
                yd.append(self.testLabels[self.testBatchNum])
                self.testBatchNum += 1
        xd = np.array(xd, dtype=np.float32)
        yd = np.array(yd, dtype=np.float32)
        return {x:xd, y:yd}
    
    def checkAgainstTestData(self, x, y, accuracy):
        accr = 0.0
        iters = 0
        try:
            while True:
                accr += accuracy.eval(feed_dict=self.getBatch(x, y, False))
                iters += 1
        except IndexError:
            self.testBatchNum = 0
            print('Test accuracy: %s ' % (100*accr/iters,))
        return 100*accr/iters
    
    
    def run(self):
        if self.dataset == 'watch':
            self.trainData, self.trainLabels, self.testData, self.testLabels = self.loadWatchData()
        else:
            sys.exit(-1)
        with tf.Session() as sess:
            maxAcc = [0, 0]
            x = tf.placeholder(dtype=tf.float32, shape=(self.batchSize, self.sequenceLength, self.features))
            y = tf.placeholder(dtype=tf.float32, shape=(None, self.categories))
            categoryProbabilities = makeModel(x, self.filtSizes, self.channels)
            with tf.name_scope('cross_entropy'):
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
            # loop over the data a number of epochs
            for j in range(self.epochs):
                try:
                    # while there is unseen data in the training set, train on the data
                    while True:
                        batch = self.getBatch(x, y, True)
                        if i%100 == 0:
                            #print(a.eval(feed_dict=batch))
                            #print(b.eval(feed_dict=batch))
                    #        print(y.eval(feed_dict=batch))
                    #        print(categoryProbabilities.eval(feed_dict=batch))
                       #     print(correct_predictions.eval(feed_dict=batch))
                            train_accuracy = accuracy.eval(feed_dict=batch)
                            print('Train Step %s: %s' % (i, train_accuracy*100))
                        train_step.run(feed_dict=batch)
                        i += 1
                except IndexError as msg:
                    self.trainBatchNum = 0
                    sys.stdout.write("Epoch " + str(j) + " ")
                    # Test accuracy calculation
                    testAcc = self.checkAgainstTestData(x, y, accuracy)
                    if testAcc > maxAcc[0]:
                        maxAcc = [testAcc, j]
            print(maxAcc)
            f = open('attempts.log', 'a')
            f.write('\n'+str(self.batchSize)+ '\t'+str(self.filtSizes)+'\t\t'+str(self.channels[1:-1])+'\t\t'+str(maxAcc[0])+'\t'+str(maxAcc[1]))
            sess.close()

def classProject():
    dataset = 'watch'
    epochs = 120
    categories = 5
    sequenceLength = 1024
    features = 3
    batchSize = 32
    filtSizes = [
            [3, 5, 7, 9, 7, 5, 3]
            ]
    channels = [
            [features, 512, 256, 128, 64, 128, 256, 512, categories]
            ]
    for i in range(len(filtSizes)):
        for j in range(5):
            run = ModelRun(dataset, filtSizes[i], channels[i], epochs, categories, sequenceLength, features, batchSize)
            run.start()
            run.join()

if __name__ == '__main__':
    classProject()
