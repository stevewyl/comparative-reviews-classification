from keras.optimizers import Adam
from reader import batch_iter
from metrics import get_callbacks
import os

class Trainer(object):

    def __init__(self, 
                 model, 
                 training_config,
                 training_mode=False,
                 checkpoint_path='./logs/',
                 save_path='',
                 lrscheduler=False,
                 tensorboard=False):

        self.model = model
        self.training_config = training_config
        self.training_mode = training_mode
        self.checkpoint_path = os.path.join(checkpoint_path, training_config.model_name)
        self.save_path = save_path
        self.lrscheduler = lrscheduler
        self.tensorboard = tensorboard

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        # 获取训练数据和验证数据
        train_steps, train_batches = batch_iter(x_train,
                                                y_train,
                                                self.training_config.batch_size)
        valid_steps, valid_batches = batch_iter(x_valid,
                                                y_valid,
                                                self.training_config.batch_size)
        # 模型训练参数
        self.model.model.compile(loss=self.training_config.loss_func,
                                 optimizer=Adam(lr=self.training_config.learning_rate),
                                 metrics=['accuracy'])
        if not self.training_mode:
            callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      tensorBoard=self.tensorboard,
                                      LRScheduler=self.lrscheduler,
                                      early_stopping=self.training_config.early_stopping,
                                      valid=(valid_steps, valid_batches))
        else:
            callbacks = get_callbacks(LRScheduler=self.lrscheduler,
                                      early_stopping=self.training_config.early_stopping,
                                      valid=(valid_steps, valid_batches))

        # 训练模型
        self.model.model.fit_generator(generator=train_batches,
                                       steps_per_epoch=train_steps,
                                       epochs=self.training_config.max_epoch,
                                       callbacks=callbacks)
        
        return self.model
