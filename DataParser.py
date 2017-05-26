#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys

desiredModel = '5datas' 
undesiredClasses = ['null', 'sit']
#classes = {'stand':1, 'sit':2, 'null':3, 'stairsup':4, 'stairsdown':5, 'bike':6, 'walk':7 }
classes = {'stand':0, 'bike':1, 'walk':2, 'stairsup':3, 'stairsdown':4, 'sit':-1, 'null':-1}
delimRow = -1*np.ones(10)

def arrayEqual(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

data = np.array(pd.read_csv(sys.argv[1], header=0))
badRowIndexes = []
lastRow = None
extraIncrement = False
lastRow = data[0, 6:]
i = 0
# filter so data only has desired model
while i < data.shape[0]:
    added = False
    if i%30000 == 0:
        print(str(100*i/data.shape[0])+ "%")
    data[i, 9] = classes[data[i, 9]]
 #   if data[i, 7] != desiredModel: 
 #       added = True
 #       badRowIndexes.append(i)
    if data[i,9] in undesiredClasses:
        badRowIndexes.append(i)
    if not arrayEqual(lastRow, data[i, 6:]) and data[i, 7] == desiredModel:
        lastRow = data[i, 6:]
        data = np.insert(data, i, delimRow, axis=0)
        extraIncrement = True
    else:
        lastRow = data[i, 6:]
#        if i%2 == 1 and added == False:
#            badRowIndexes.append(i)
    if extraIncrement:
        extraIncrement = False
        i += 1
    i += 1
data = np.delete(data, 8, 1)    # delete device
data = np.delete(data, 7, 1)    # delete device type
data = np.delete(data, 6, 1)    # delete user
data = np.delete(data, 2, 1)    # delete creation_time
data = np.delete(data, 1, 1)
data = np.delete(data, 0, 1)    # delete index
# reverse it so when looping to delete, it doesn't change the values of the rows
badRowIndexes.reverse()
data = np.delete(data, badRowIndexes, 0)
d = data[:,:-1]
data[:,:-1] = (d-d.min(0))/d.ptp(0)
print("Removed undesired rows")
np.savetxt(desiredModel+'.csv', np.array(data, dtype=np.float32))
