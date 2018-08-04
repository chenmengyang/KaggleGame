import os
import csv
import numpy as np

# read the train data, return sentences words indexes, and lables
def readTrain(word_to_index):
    trainingSet = []
    trainingLabels = []
    with open('./data/train.tsv') as f:
        f_tsv = csv.DictReader(f, delimiter='\t')
        errors = 0
        for row in f_tsv:
            try:
                indexes = [word_to_index[word] for word in row['Phrase'].lower().strip().split()]
                trainingSet.append(np.array(indexes))
                trainingLabels.append(row['Sentiment'])
            except:
                errors += 1
                # print(row['Phrase'])
        # print ('there are {} number of error texts'.format(errors))
    return trainingSet, trainingLabels

# read the test data, return sentences words indexes
def readTest():
    testSet = []
    with open('./data/test.tsv') as f:
        f_tsv = csv.DictReader(f, delimiter='\t')
        errors = 0
        for row in f_tsv:
            try:
                indexes = [word_to_index[word] for word in row['Phrase'].lower().strip().split()]
                testSet.append(np.array(indexes))
            except:
                errors += 1
                # print(row['Phrase'])
        print ('there are {} number of error sentences in test set'.format(errors))
    return testSet

# read the pretrained word-embedding matrix
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

# get the maximun length of list of train and test sentences
def getMaxLen(trainData, setData):
    max1 = max([len(x) for x in trainData])
    max2 = max([len(x) for x in setData])
    return max1 if max1>=max2 else max2

# 
def applyPadding(maxlen, index, dataset):
    newSet = []
    for arr in dataset:
        if len(arr) < maxlen:
            arr = np.pad(arr, (0, maxlen-len(arr)), 'constant')
        newSet.append(arr)
    return newSet

# lets load the word embedding matrix first
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./data/glove.6B.50d.txt')
# load training set, of course we will convert words into embeddings
train, _ = readTrain(word_to_index)
# well, do the same thing the test data
test = readTest()
# print ('length of training set is {}'.format(len(train)))
# print ('the first training sample length is {}'.format(len(train[0])))
# now find the maximun length of the training set and the test set
maxlen = getMaxLen(train, test)
# print ('maximun sentence length is {}'.format(maxlen))
# apply zero padding values to train and test set
# great man! we are done, this is the thing we will feed to the neural network, boom!
train = applyPadding(maxlen, 0, train)
test = applyPadding(maxlen, 0, test)
print ('preview your training set, the frist train data had been converted into {}'.format(train[0]))