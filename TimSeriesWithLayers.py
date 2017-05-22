#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

sequenceLength = 1024
features = 6
batchSize = 64

class DataParser:
    testX = None
    testY = None

    path = ""

    def DataParser(self):
        raw = np.array(pd.read_csv(path, delimiter='\t',header=-1))
        columnNames = raw[0,:]
        #TODO

def getParameters(din):
    i = tf.ones((din.shape[0], 1))
    ave = tf.matmul(tf.transpose(din), i)/din.shape[0]
    cov = tf.norm(tf.subtract(din, tf.transpose(ave)))/din.shape[0]
    return ave, cov

def BatchNorm(din):
    # scale and offest should be tf.Variables
    # treat them like weights and bias, train them
    ave, cov = getParameters(din, ave)
    return tf.nn.batch_normalization(din, ave, cov, 0, 1, 1e-7)

def makeModel(X, Y):
    kernels = [11, 8, 5]
    filters = [sequenceLength/2, sequenceLength/4, sequenceLength/8]
    layers = [X]
    for i in range(len(filters)):
        conv = tf.layers.conv1d(layers[i], filters[i], kernels[i])
        bnorm = BatchNorm(conv)
        layers.append(tf.nn.relu(bnorm))
    layers.append(GlobalBatchNorm(layers[-1]))
    layers.append(tf.nn.softmax_cross_entropy_with_logits(logits=layers[-1], labels=Y))
    return layers

if __name__ == "__main__":

    X = tf.placeholder(tf.float32, [None, sequenceLength, features])
    Y = tf.placeholder(tf.float32, [None, 1])
    model = makeModel(X,Y)
    ypred = model[-1]
    crossEntropy = tf.reduce_mean(ypred)
    trainStep = tf.train.AdamOptimizer(0.01).minimize(crossEntropy)
    correctPrediction = tf.equal(tf.argmax(ypred, 1), tf.argmax(batchY, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    
    dataParser = DataParser()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        epoch = 0
        while dataParser.hasNext:
            batchX, batchY = dataParser.next(1000)
            feed_dict={X:batchX, Y:batchY}
            if epoch%100 == 0:
                train_accuracy = accuracy.eval(feed_dict)
                print("Epoch %d, training accuracy %g"%(epoch, train_accuracy))
            sess.run(trainStep, feed_dict=feed_dict)
        print("test accuracy %g"%accuracy.eval(feed_dict={X:dtatParser.testX, Y:dataParser.testY}))

