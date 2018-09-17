from metrices import my_iou_metric
from losses import bce_dice_loss, dice_loss
import tensorflow as tf
from tensorflow.python.keras import optimizers

def train(model, epochs, batch_size, trainX, trainY, valX, valY):
    model.compile(loss=bce_dice_loss, optimizer=optimizers.adam(lr=0.01), metrics=[my_iou_metric, dice_loss])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_my_iou_metric',mode='max',patience=20, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("./data/keras.model", monitor='val_my_iou_metric', mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.5, patience=4, min_lr=0.0001, verbose=1)

    history = model.fit(
        x=trainX,
        y=trainY,
        validation_data=[valX, valY],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=2
    )

    return history, model