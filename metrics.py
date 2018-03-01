from pathlib import Path
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from sklearn.metrics import confusion_matrix
import math

# 学习率衰减策略
# 每两个epoch学习率减少 0.5^(epoch+1/2)
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch) / epochs_drop))
    return lrate

class F1score(Callback):

    def __init__(self, valid_steps, valid_batches):
        super(F1score, self).__init__()
        self.valid_steps = valid_steps
        self.valid_batches = valid_batches

    # 每一个epoch计算一次f1分数
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
        for i in range(f1.shape[0]):
            print(i, ' - f1: {:04.2f}\n'.format(f1[i] * 100))
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

def get_callbacks(log_dir=None, valid=(), tensorBoard=False, 
                  early_stopping=True, LRScheduler=False):

    callbacks = []
    
    if log_dir and tensorBoard:
        if not Path(log_dir).exists():
            print('Successfully made a directory: {}'.format(log_dir))
            Path(log_dir).mkdir()
        callbacks.append(TensorBoard(log_dir=log_dir,                             
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_grads=True))

    if valid:
        callbacks.append(F1score(*valid))

    if log_dir:
        if not Path(log_dir).exists():
            print('Successfully made a directory: {}'.format(log_dir))
            Path(log_dir).mkdir()

        file_name = '_'.join(['model_weights', '{epoch:02d}', '{f1:2.2f}']) + '.h5'
        save_callback = ModelCheckpoint(Path(log_dir) / file_name,
                                        monitor='val_acc',
                                        verbose=1,
                                        save_best_only=True)
        callbacks.append(save_callback)

    if early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=2, mode='max'))
    
    if LRScheduler:
        callbacks.append(LearningRateScheduler(step_decay))

    return callbacks



