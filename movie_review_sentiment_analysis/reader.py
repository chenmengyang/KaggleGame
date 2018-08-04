import os
import csv
# from collections import namedtuple

def readTrain():
    trainingSet = []
    trainingLabels = []
    with open('./data/train.tsv') as f:
        f_tsv = csv.DictReader(f, delimiter='\t')
        for row in f_tsv:
            trainingSet.append(row['Phrase'])
            trainingLabels.append(row['Sentiment'])
    return trainingSet, trainingLabels

def readTest():
    testSet = []
    with open('./data/test.tsv') as f:
        f_tsv = csv.DictReader(f, delimiter='\t')
        for row in f_tsv:
            testSet.append(row['Phrase'])
    return testSet

train, _ = readTrain()
test = readTest()
print ('length of training set is {}'.format(len(train)))
print (train[0])
print ('length of test set is {}'.format(len(test)))
print (test[0])