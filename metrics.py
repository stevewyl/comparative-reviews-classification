import numpy as numpy
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler

def get_callbacks(log_dir=None, valid=(), tensorBoard=False, eary_stopping=True):

    callbacks = []

    

    if eary_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=2, mode='max'))

    return callbacks
