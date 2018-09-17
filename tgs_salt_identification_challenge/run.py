import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
# from PIL import Image
plt.style.use('seaborn-white')
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
from skimage.transform import resize
from datetime import datetime
import random

# 1 import data
trainX, trainY, valX, valY, testX = salt_dataset()
print ('finish step 1')

# 2 import model
unet_model = Model()
print ('finish step 2')

# 3 train
his, unet_model = train(unet_model, 200, 32, trainX, trainY, valX, valY)
print ('finish step 3')

# 4 visualise training history
ioudf = pd.DataFrame(data={
    'iou': his.history['my_iou_metric'],
    'val_iou': his.history['val_my_iou_metric'],
})
lossdf = pd.DataFrame(data={
    'loss': his.history['loss'],
    'val_loss': his.history['val_loss'],
})
dlossdf = pd.DataFrame(data={
    'dice_loss': his.history['dice_loss'],
    'val_dice_loss': his.history['val_dice_loss'],
})
try:
    ioudf.plot.line()
    plt.savefig('./data/figures/train_iou.png')
    lossdf.plot.line()
    plt.savefig('./data/figures/train_loss.png')
    dlossdf.plot.line()
    plt.savefig('./data/figures/train_dice_loss.png')
except:
    pass

# 5 find the best_threshold
testY1 = predict(unet_model, valX, valY)

# 6 submit result
sub = pd.read_csv('./data/sample_submission.csv')
def rle_encode(im):
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

sub['rle_mask'] = [rle_encode(imx) for imx in testY1]
sub.to_csv('./data/outputs/sub_{}.csv'.format(str(datetime.today())), index=False)

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
    plt.savefig('./data/figures/final_review.png')
except:
    pass