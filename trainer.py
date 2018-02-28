from keras.optimizers import Adam
from reader import batch_iter
from metrics import get_callbacks

class Trainer(object):

    def __init__(self, 
                 model, 
                 training_config,
                 checkpoint_path='', 
                 save_path='',
                 tensorboard=False):

        self.model = model
        self.training_config = training_config
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
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
        self.model.compile(loss=self.training_config.loss_func,
                           optimizers=Adam(lr=self.training_config.learning_rate),
                           metric=['accuracy'])
        callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                  tensorBoard=self.tensorboard,
                                  eary_stopping=self.training_config.early_stopping,
                                  valid=(valid_steps, valid_batches))
        # 训练模型
        self.model.fit_generator(generator=train_batches,
                                 steps_per_epoch=train_steps,
                                 epochs=self.training_config.max_epoch,
                                 callbacks=callbacks)
        
