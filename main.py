import re
import time
import pandas as pd
import numpy as np
import os

from reader import load_data_and_labels, load_embeddings
from trainer import Trainer
from model_library import TextCNNBN, TextInception, convRNN, HAN, MHAN, SelfAtt, fasttext, Bi_RNN
from config import ModelConfig, TrainingConfig
from utils import customed_heatmap, read_line_data, split_sent

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def get_sequences(tokenizer, train_data, mode, max_length):
    if mode == 'seq':
        res = []
        for sent in train_data:
            word_ids = tokenizer.texts_to_sequences(sent)
            padded_seqs = pad_sequences(word_ids, maxlen=max_length)
            res.append(padded_seqs)
        return np.asarray(res)
    elif mode == 'text':
        word_ids = tokenizer.texts_to_sequences(train_data)
        padded_seqs = pad_sequences(word_ids, maxlen=max_length)
        return padded_seqs

# 一些控制流程的全局变量
MAX_WORDS = 100
MAX_SENTS = 5
MODEL_NAME = 'HAN'
CUT_MODE = 'simple'
TEXT_FORMAT = 'text'
TEST_SIZE = 0.2
N_FOLDS = 10
FOLDS_EPOCHS = 1
CV = False #是否进行交叉验证
CHECK_HIDDEN = False #是否检查隐性比较句的错误情况
ATTENTION_V = False #是否可视化attention权重
PREDICT = False #是否预测新评论

# 读入数据
sents, labels, _ = load_data_and_labels('./data/jd_comp_final_v5.xlsx', ['not_hidden', 'non'], 'word')
labels = to_categorical(np.array(labels))
if PREDICT:
    df_predict = pd.read_excel('./data/jd_20w_v2.xlsx')
    predict_text = df_predict['segment'].tolist()
    sents = sents + predict_text

# 初始化文本->index
tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower = True, split = " ")
tokenizer.fit_on_texts(sents)
vocab = tokenizer.word_index
vocab['UNK'] = 0
word2idx = {v:k for k,v in vocab.items()}

# 加载词向量文件
EMBED_FILE = './data/embeddings/word_vectors_256d_20171228_5.txt'
if EMBED_FILE and MODEL_NAME != 'one-hot':
    print('loading word embeddings...')
    EMBED_TYPE = re.findall(r'(?<=/)\w+(?=_v)', EMBED_FILE)[0]
    EMBED_DIMS = int(re.findall(r'(?<=_)\d+(?=d)', EMBED_FILE)[0])
    embedding_matrix = load_embeddings(EMBED_FILE, vocab, EMBED_DIMS)
else:
    EMBED_TYPE = 'scratch'
    EMBED_DIMS = 256
    embedding_matrix = None

# 模型及权重保存路径
DATE = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
model_file = './model/' + MODEL_NAME + '_' + DATE + '.yaml'
weight_file = './model/' + MODEL_NAME + '_weights_' + DATE + '.hdf5'

# HAN族模型需要的文本输入格式
if MODEL_NAME in ['HAN','MHAN']:
    print('prepare inputs for HAN series model...')
    if EMBED_TYPE == 'word' or 'scratch':
        MAX_WORDS = 20
        MAX_SENTS = 5
    elif EMBED_TYPE == 'char':
        MAX_WORDS = 30
        MAX_SENTS = 6
    N_LIMIT = MAX_WORDS * MAX_SENTS
    x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in sents]
    if PREDICT:
        p_x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in predict_text]
    TEXT_FORMAT = 'seq'
    new_name = MODEL_NAME + '_' + str(MAX_WORDS) + '_' + str(MAX_SENTS)
    model_file = './model/' + new_name + '_' + DATE + '.yaml'
    weight_file = './model/' + new_name + '_weights_' + DATE + '.hdf5'

# 初始化参数设置
model_cfg = ModelConfig(MAX_WORDS, MAX_SENTS, EMBED_DIMS, len(vocab)+1, MODEL_NAME, ntags=2)
train_cfg = TrainingConfig(ntags=2)

# 初始化模型
# 单次训练还是交叉验证训练
print(EMBED_TYPE + ' model ' + MODEL_NAME + ' start training...')

# 文本index化
data = get_sequences(tokenizer, x, TEXT_FORMAT, MAX_WORDS)
    
# 初始化模型
model = HAN(model_cfg, embedding_matrix)
model.model.summary()
if model_file:
    model.plot(os.path.join('./model', MODEL_NAME+'.jpg'))
    model.save_model(model_file)

trainer = Trainer(model, train_cfg)

if __name__ == '__main__':
    if not CV:
        x_train, x_valid, y_train, y_valid = train_test_split(
            data, labels, test_size=TEST_SIZE)
        trainer.train(x_train, y_train, x_valid, y_valid)

