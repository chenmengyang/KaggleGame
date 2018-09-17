import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from skimage.transform import resize
path = './data/'

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i

def salt_dataset():
    trainX = [resize(img_to_array(load_img(path + 'train/images/{}'.format(filename), grayscale=True))/255, (128,128)) for filename in sorted(os.listdir(path + 'train/images'))]
    trainY = [resize(img_to_array(load_img(path + 'train/masks/{}'.format(filename), grayscale=True))/255, (128,128)) for filename in sorted(os.listdir(path + 'train/masks'))]
    coverClass = [cov_to_class(np.sum(msk)/(128*128)) for msk in trainY]
    traindf = pd.DataFrame(data={
        'X': trainX,
        'Y': trainY,
        'coverClass': coverClass
    }, index=sorted(os.listdir(path + 'train/images')))
    trainX, valX, trainY, valY = train_test_split(
        traindf['X'],
        traindf['Y'],
        test_size = 0.2,
        stratify=traindf.coverClass
    )
    trainX = np.append(list(trainX), [np.fliplr(imgx) for imgx in trainX], axis=0)
    trainY = np.append(list(trainY), [np.fliplr(imgx) for imgx in trainY], axis=0)
    valX = np.array(list(valX))
    valY = np.array(list(valY))
    ts = pd.read_csv(path + 'sample_submission.csv')
    testX = np.array([resize(img_to_array(load_img(path + 'test/images/{}.png'.format(filename), grayscale=True))/255, (128,128)) for filename in ts['id']])
    return trainX, trainY, valX, valY, testX