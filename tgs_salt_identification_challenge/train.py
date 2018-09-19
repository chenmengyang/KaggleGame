from metrices import my_iou_metric
from losses import bce_dice_loss, dice_loss
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras import optimizers

def train(model, epochs, batch_size, trainX, trainY, valX, valY):
    model.compile(loss=bce_dice_loss, optimizer=optimizers.Adam(lr=0.01), metrics=[my_iou_metric, bce_dice_loss, 'acc'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_my_iou_metric',mode='max',patience=20, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("./data/keras.model", monitor='val_my_iou_metric', mode='max', save_best_only=True, verbose=1)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=4, min_lr=0.00001, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max', factor=0.5, patience=4, min_lr=0.00001, verbose=1)

    history = model.fit(
        x=trainX,
        y=trainY,
        validation_data=[valX, valY],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=2
    )

    # visualise training history, save images to data folder
    try:
        ioudf = pd.DataFrame(data={
            'iou': history.history['my_iou_metric'],
            'val_iou': history.history['val_my_iou_metric'],
        })
        lossdf = pd.DataFrame(data={
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
        })
        accdf = pd.DataFrame(data={
            'acc': history.history['acc'],
            'val_acc': history.history['val_acc'],
        })
        dlossdf = pd.DataFrame(data={
            'bce_dice_loss': history.history['bce_dice_loss'],
            'val_bce_dice_loss': history.history['val_bce_dice_loss'],
        })
        ioudf.plot.line()
        plt.savefig('./data/figures/train_iou.png')
        # plt.show()
        lossdf.plot.line()
        plt.savefig('./data/figures/train_loss.png')
        # plt.show()
        dlossdf.plot.line()
        plt.savefig('./data/figures/train_dice_loss.png')
        # plt.show()
        accdf.plot.line()
        plt.savefig('./data/figures/train_accuracy.png')
        # plt.show()
    except Exception as e:
        print ('save training images error')

    return history, model