import math
import numpy as np

trainSplit = .2
numOfCategories = 5

def convertToDictionary(data, labels):
    dataDict = {}
    oldLabel = -1
    instForCat = []
    # notice this will omit the delimitor from being added to the dictionary
    for i in range(len(labels)):
        if oldLabel != labels[i]:
            if len(instForCat) != 0:
                if oldLabel not in dataDict.keys():
                    dataDict[oldLabel] = [instForCat]
                else:
                    dataDict[oldLabel].append(instForCat)
                instForCat = []
            else:
                instForCat.append(data[i])
        else:
            instForCat.append(data[i])
        oldLabel = labels[i]
    return dataDict

def getData(datapath):
    data = []
    f = open(datapath)
    for line in f:
        a = []
        words = line.split(' ')
        for word in words:
            a.append(word)
        data.append(a)
    data = np.array(data, dtype=np.float32)
    f.close()
    return data[:,:-1], data[:,-1]

def splitData(dataDict):
    testDict = {}
    for k in dataDict.keys():
        testList = []
        numOfTest = math.floor(len(dataDict[k])*trainSplit)
        for i in range(numOfTest):
            testList.append(dataDict[k].pop())
        testDict[k] = testList
    return dataDict, testDict

# takes a list of instances for a single category
# returns a list of batches for that category
def getBatches(instances, sequenceLength):
    batches = []
    i = 0
    while i < math.floor(len(instances)/sequenceLength)-1:
        batches.append(instances[sequenceLength*i:sequenceLength*(i+1)])
        i += 1
    return batches

# returns a list of one-hot encoded labels with len = numOfEncodings
def encodeLabels(cat, numOfEncodings):
    encodings = []
    a = np.zeros(numOfCategories, dtype=np.float32)
    a[int(cat)] = 1
    for i in range(numOfEncodings):
        encodings.append(a)
    return encodings

'''def convertToLists(dataDict, sequenceLength):
    batches = []
    encodedLabels = []
    for k in dataDict.keys():
        for run in dataDict[k]:
            b = getBatches(run, sequenceLength)
            encodedLabels.extend(encodeLabels(k, len(b)))
            batches.extend(b)
    return batches, encodedLabels
'''

def convertToLists(dataDict, sequenceLength):
    batches = []
    encodedLabels = []
    for k in dataDict.keys():
        for run in dataDict[k]:
            b = np.array(getBatches(run, sequenceLength))
            if len(b) != 0:
                encodedLabels.extend(encodeLabels(k, len(b)))
                batches.extend(b)
    # shuffle the data so that categories are not next to each other
    c = np.array([batches, encodedLabels]).T
    np.random.shuffle(c)
    b = np.array(c[:,0])
    e = np.array(c[:,1])
    batches = np.ndarray(shape=(len(batches), 1024, 3), dtype=np.float32)
    encodedLabels = np.ndarray(shape=(len(encodedLabels), numOfCategories), dtype=np.float32)
    for i in range(len(batches)):
        batches[i] = b[i]
        encodedLabels[i] = e[i]
    return batches, encodedLabels

def loadData():
    data, labels = getData()
    dataDict = convertToDictionary(data, labels)
    trainDict, testDict = splitData(dataDict)
    trainData, trainLabels = convertToLists(trainDict)
    testData, testLabels = convertToLists(testDict)
    return trainData, trainLabels, testData, testLabels
