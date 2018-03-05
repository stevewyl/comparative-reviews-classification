import os
from pathlib import Path
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
import keras.backend as K
from sklearn.metrics import confusion_matrix
import math
import matplotlib.pyplot as plt

# 学习率衰减策略
# 每两个epoch学习率减少 0.5^(epoch+1/2)
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    return LearningRateScheduler(schedule)

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
            print('\t')
            print(i, '- f1: {:04.2f}'.format(f1[i] * 100))
        logs['f1'] = f1[0]

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

class LRFinder(Callback):

    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.
    # Usage

        ```python
           lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=10, epochs=3)
           model.fit(X_train, Y_train, callbacks=[lr_finder])

           lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References

        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr-self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')

    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')


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

        file_name = '_'.join(['model_weights', '{epoch:02d}', '{acc:2.4f}']) + '.h5'
        save_callback = ModelCheckpoint(os.path.join(log_dir, file_name),
                                        monitor='acc',
                                        verbose=1,
                                        save_best_only=True)
        callbacks.append(save_callback)

    if early_stopping:
        callbacks.append(EarlyStopping(monitor='f1', patience=2, mode='max'))
    
    if LRScheduler:
        callbacks.append(LearningRateScheduler(step_decay_schedule(0.001, 0.75, 2)))

    return callbacks



