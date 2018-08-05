import pandas as pd 
train = pd.read_csv('./data/train.tsv', sep='\t')
test = pd.read_csv('./data/test.tsv', sep='\t')
import csv
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text as T

# give C =5 , convert Y=1 into [0,1,0,0,0]
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def getIndex(word, word_to_index):
    idx = 0
    try:
        idx = word_to_index[word]
    except:
        idx = 0
    return idx

# read the train data, return sentences words indexes, and lables
def readTrain(word_to_index):
    '''texts = ' '.join(list(train['Phrase'].values))
                rst = T.text_to_word_sequence(texts,
                                                           filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                                           lower=True,
                                                           split=" ")
                #print(rst)
                texts = list(train['Phrase'].values)
                tokenizer = Tokenizer(num_words=None) #num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
                tokenizer.fit_on_texts(texts)
                print( tokenizer.word_counts) #[('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)]
                print( tokenizer.word_index) #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
                print( tokenizer.word_docs) #{'some': 2, 'thing': 2, 'to': 2, 'drink': 1,  'eat': 1}
                print( tokenizer.index_docs)
                word_index = tokenizer.word_index
                print('Found %s unique tokens.' % len(word_index))'''
    trainingSet = []
    trainingLabels = []
    errors = 0
    # for word in T.text_to_word_sequence(train['Phrase'].get(0),
    #                                         filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
    #                                         lower=True,
    #                                         split=" "):
    # 	print(word_to_index[word])
    for i,row in enumerate(zip(train['PhraseId'].values, train['Phrase'].values)):
        try:
            # debugging
            # if (i == 10):
            #     # print ('fuck')
            #     print (row)
            #     print (T.text_to_word_sequence(row,
            #                                    filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
            #                                    lower=True,
            #                                    split=" "))
            #     break
            # 
            indexes = [getIndex(word, word_to_index) for word in T.text_to_word_sequence(row[1],
                                               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")]
            trainingSet.append(indexes)
            trainingLabels.append(int(train['Sentiment'].get(i)))
        except  Exception as e:
            errors += 1
            # trainErrors.writerow(row)
            # print(e)
    print ('there are {} number of error texts'.format(errors))
    return trainingSet, convert_to_one_hot(np.asarray(trainingLabels, dtype=int), 5)

# read the test data, return sentences words indexes
def readTest(word_to_index):
    testSet = []
    #with open('./data/test.tsv') as f:
    f_tsv = test
    errors = 0
    for row in zip(test['PhraseId'].values, test['Phrase'].values):
        try:
            indexes = [getIndex(word, word_to_index) for word in T.text_to_word_sequence(row[1],
                                               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")]
            #print(indexes)
            testSet.append(indexes)
        except:
            errors += 1
            # testErrors.writerow(row)
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

# apply 0 indexes to sentences, for making all sentences have the same length
def applyPadding(maxlen, index, dataset):
    newSet = []
    for arr in dataset:
        if len(arr) < maxlen:
            arr = np.pad(arr, (0, maxlen-len(arr)), 'constant')
        newSet.append(arr)
    return np.asarray(newSet)

def load_data():
    # lets load the word embedding matrix first
    word_to_index, _, word_to_vec_map = read_glove_vecs('./data/glove.6B.50d.txt')
    # load train and test data into word indexes first
    trainX, trainY = readTrain(word_to_index)
    test = readTest(word_to_index)
    # print ('length of training set is {}'.format(len(train)))
    # print ('the first training sample length is {}'.format(len(train[0])))
    # now find the longest sentence length of the training set and the test set
    maxlen = getMaxLen(trainX, test)
    # print ('maximun sentence length is {}'.format(maxlen))
    # apply zero padding values to train and test set
    # great man! we are done, this is the thing we will feed to the neural network, boom!
    trainX = applyPadding(maxlen, 0, trainX)
    test = applyPadding(maxlen, 0, test)
    # print ('preview your training set, the frist train data had been converted into {}'.format(train[0]))
    return trainX, trainY, test, (word_to_index, word_to_vec_map), maxlen


# word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./data/glove.6B.50d.txt')
# f1 = open('./data/trainError.csv', 'w')
# trainErrors = csv.writer(f1)
# trainX, trainY = readTrain(word_to_index)
# f1.close()

# f2 = open('./data/testError.csv', 'w')
# testErrors = csv.writer(f2)
# test = readTest(word_to_index)
# f2.close()