import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
plt.style.use('seaborn-white')
plt.switch_backend('agg')
import seaborn as sns
sns.set_style("white")
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from data import salt_dataset
from models.unet import Model
from train import train
from predict import predict
from metrices import my_iou_metric1
from datetime import datetime
import logging
import random

if __name__ == '__main__':
    # 0 config logging
    logging.basicConfig(
        filename='./data/outputs/logs/log-{}.txt'.format(str(datetime.today())),
        level=logging.INFO
    )

    # 1 import data
    trainX, trainY, valX, valY, testX = salt_dataset()
    logging.info('finish step 1')

    # 2 import model
    unet_model = Model()
    logging.info('finish step 2')

    # 3 train
    his, unet_model = train(unet_model, 200, 32, trainX, trainY, valX, valY)
    logging.info('finish step 3')

    # 4 find the best_threshold on val set and do the predict on test set
    testY1 = predict(unet_model, valX, valY, testX)

    # 5 submit result
    sub = pd.read_csv('./data/sample_submission.csv')
    def rle_encode(im):
        pixels = im.flatten(order = 'F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    sub['rle_mask'] = [rle_encode(imx) for imx in testY1]
    sub.to_csv('./data/outputs/sub_{}.csv'.format(str(datetime.today())), index=False)

    # visualise traning result randomly
    try:
        n = testY1.shape[0]
        m = 10
        inds = random.sample([i for i in range(n)], m)
        fig, axs = plt.subplots(m, 2, figsize=(4,2*m))
        for index, i in enumerate(inds):
            axs[index, 0].imshow(testY1[i])
            axs[index, 0].set_title('pred', fontsize=15)
            axs[index, 1].imshow(testX[i].reshape((128,128)) * 255)
            axs[index, 1].set_title('gt-input', fontsize=15)
        plt.subplots_adjust(hspace=.5)
        plt.savefig('./data/outputs/figures/final_review.png')
    except Exception as e:
        logging.error('saving sample review image error')