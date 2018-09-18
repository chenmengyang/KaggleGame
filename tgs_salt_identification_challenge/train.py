from metrices import my_iou_metric
from losses import bce_dice_loss, dice_loss
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras import optimizers

def train(model, epochs, batch_size, trainX, trainY, valX, valY):
    model.compile(loss=bce_dice_loss, optimizer=optimizers.Adam(lr=0.01), metrics=[my_iou_metric, bce_dice_loss])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_my_iou_metric',mode='max',patience=20, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("./data/keras.model", monitor='val_bce_dice_loss', save_best_only=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_bce_dice_loss', factor=0.5, patience=4, min_lr=0.0001, verbose=1)

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
    ioudf = pd.DataFrame(data={
        'iou': history.history['my_iou_metric'],
        'val_iou': history.history['val_my_iou_metric'],
    })
    lossdf = pd.DataFrame(data={
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
    })
    dlossdf = pd.DataFrame(data={
        'bce_dice_loss': history.history['bce_dice_loss'],
        'val_bce_dice_loss': history.history['val_bce_dice_loss'],
    })
    try:
        ioudf.plot.line()
        plt.savefig('./data/figures/train_iou.png')
        lossdf.plot.line()
        plt.savefig('./data/figures/train_loss.png')
        dlossdf.plot.line()
        plt.savefig('./data/figures/train_dice_loss.png')
    except Exception as e:
        print ('save training images error')

    return history, model