import json


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self, max_words, embed_size, vocab_cnt,
                 model_name, ntags=2):
        self.max_words = max_words #句子序列最大长度
        self.embed_size = embed_size #词向量维度
        self.vocab_cnt = vocab_cnt #词语总数
        self.drop_rate = [0.5,0.3] #丢弃率
        self.re_drop = [0.25,0.15] #丢弃率（rnn）
        self.fc_units = [64] #fc层输出维度
        self.rnn_units = [256,128] #rnn层输出维度
        self.activation_func = 'relu' #激活函数
        self.classifier = 'sigmoid' #分类器
        self.loss_func = 'binary_crossentropy' # 损失函数
        self.pool_size = [5,5,5]  #池化层步长
        self.conv_size = [192, 128] #卷积层channel数
        self.ntags = ntags
        self.model_name = model_name

        if self.ntags > 2:
            self.classifier = 'softmax'
            self.loss_func = 'categorical_crossentropy'

        if model_name == 'HAN':
            self.max_sents = 5
            self.att_size = [100, 50]
        elif model_name == 'MHAN':
            self.max_sents = 5
            self.ws1 = [100, 50]
            self.r = [4,2]
        elif model_name == 'Self_Att':
            self.ws1 = 300
            self.r = 10
        elif model_name == 'TextCNN':
            self.conv_size = 128
            self.filter_size = [3,4,5]
        elif model_name == 'TextCNNBN':
            self.filter_size = [1,2,3,4,5]
        elif model_name == 'inception':
            self.filter_size = [[1],[1,3],[3,5],[3]]
        elif model_name == 'convRNN':
            self.filter_size = 3
        else:
            print('This model does not exist in model_library.py')

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(vars(self), f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, file):
        with open(file) as f:
            variables = json.load(f)
            self = cls()
            for key, val in variables.items():
                setattr(self, key, val)
        return self


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self, batch_size=64, optimizer='adam', learning_rate=0.001, lr_decay=0.9,
                 clip_gradients=5.0, max_epoch=10, early_stopping=True, patience=2,
                 train_embeddings=False, max_checkpoints_to_keep=5):

        # Batch size
        self.batch_size = batch_size

        # Optimizer for training the model.
        self.optimizer = optimizer

        # Learning rate for the initial phase of training.
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

        # If not None, clip gradients to this value.
        self.clip_gradients = clip_gradients

        # The number of max epoch size
        self.max_epoch = max_epoch

        # Parameters for early stopping
        self.early_stopping = early_stopping
        self.patience = patience

        # Fine-tune word embeddings
        self.train_embeddings = train_embeddings

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
