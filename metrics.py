from pathlib import Path
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from sklearn.metrics import confusion_matrix

class F1score(Callback):

    def __init__(self, valid_steps, valid_batches):
        super(F1score, self).__init__()
        self.valid_steps = valid_steps
        self.valid_batches = valid_batches

    def on_epoch_end(self, epoch, logs={}):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true = label
            y_true = np.argmax(y_true, -1)
            y_pred = self.model.predict_on_batch(data)
            y_pred = np.argmax(y_pred, -1)

            a, b, c = self.count_correct_and_pred(y_true, y_pred)
            correct_preds += a
            total_preds += b
            total_correct += c

        f1 = self._calc_f1(correct_preds, total_correct, total_preds)
        print(' - f1: {:04.2f}'.format(f1 * 100))
        logs['f1'] = f1

    def _calc_f1(self, correct_preds, total_correct, total_preds):
        p = correct_preds / total_preds 
        r = correct_preds / total_correct 
        f1 = 2 * p * r / (p + r)
        return f1

    def count_correct_and_pred(self, y_true, y_pred):
        confusion_mat = confusion_matrix(y_true, y_pred)
        correct_preds = np.diagonal(confusion_mat)
        total_preds = np.sum(confusion_mat.T, -1)
        total_correct = np.sum(confusion_mat, 1)
        return correct_preds, total_correct, total_preds

def get_callbacks(log_dir=None, valid=(), tensorBoard=False, eary_stopping=True):

    callbacks = []
    
    if log_dir and tensorBoard:
        if not Path(log_dir).exists():
            print('Successfully made a directory: {}'.format(log_dir))
            Path(log_dir).mkdir()
    callbacks.append(TensorBoard(log_dir))

    if valid:
        callbacks.append(F1score(*valid))

    if log_dir:
        if not Path(log_dir).exists():
            print('Successfully made a directory: {}'.format(log_dir))
            Path(log_dir).mkdir()

        file_name = '_'.join(['model_weights', '{epoch:02d}', '{f1:2.2f}']) + '.h5'
        save_callback = ModelCheckpoint(Path(log_dir) / file_name,
                                        monitor='f1',
                                        save_weights_only=True)
        callbacks.append(save_callback)

    if eary_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=2, mode='max'))

    return callbacks



