import re
import time
import pandas as pd
import numpy as np
import os
import sys

from reader import load_data_and_labels, load_embeddings
from trainer import Trainer
from evaluator import Evaluator
from model_library import TextCNNBN, TextInception, convRNN, HAN, MHAN, SelfAtt, fasttext, Bi_RNN
from config import ModelConfig, TrainingConfig
from visualization import visualize_attention
from utils import split_sent, read_line_data

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

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
CHECK_HIDDEN = False #是否检查隐性比较句的错误情况
ATTENTION_V = True #是否可视化attention权重
PREDICT = False
RAND = True
LABEL = 0

# 读入数据
sents, labels, _ = load_data_and_labels('./data/jd_comp_final_v5.xlsx', ['not_hidden', 'non'], 'word')
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
    sents = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in sents]
    if PREDICT:
        p_x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in predict_text]
    TEXT_FORMAT = 'seq'
    new_name = MODEL_NAME + '_' + str(MAX_WORDS) + '_' + str(MAX_SENTS)
    model_file = './model/' + new_name + '_' + DATE + '.yaml'
    weight_file = './model/' + new_name + '_weights_' + DATE + '.hdf5'

# 初始化参数设置
model_cfg = ModelConfig(MAX_WORDS, MAX_SENTS, EMBED_DIMS, len(vocab)+1, MODEL_NAME, ntags=2)
train_cfg = TrainingConfig(ntags=2, model_name=MODEL_NAME)

# 初始化模型
# 单次训练还是交叉验证训练
print(EMBED_TYPE + ' model ' + MODEL_NAME + ' start training...')

# 文本index化
data = get_sequences(tokenizer, sents, TEXT_FORMAT, MAX_WORDS)
# 可视化文本预处理
if ATTENTION_V and MODEL_NAME in ['HAN', 'Self_Att', 'MHAN']:
    if not RAND:
        show_text = read_line_data('./data/reviews_example.txt')
        if MODEL_NAME in ['HAN', 'MHAN']:
            show_text = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in show_text]
        SHOW_TEXT = get_sequences(tokenizer, show_text, TEXT_FORMAT, MAX_WORDS)
    else:
        x_samples = np.array([data[k] for k,v in enumerate(labels) if v == LABEL])
        random_index = np.random.randint(x_samples.shape[0], size = 15)
        SHOW_TEXT = x_samples[random_index]

def training(data, labels, model_file, model_cfg, train_cfg, cv=False):
    if cv is not True:
        # 初始化模型
        model = HAN(model_cfg, embedding_matrix)
        model.model.summary()
        if model_file:
            #model.plot(os.path.join('./model', MODEL_NAME+'.jpg'))
            model.save_model(model_file)
        print('single training')
        trainer = Trainer(model, train_cfg)
        labels = to_categorical(labels)
        x_train, x_valid, y_train, y_valid = train_test_split(
            data, labels, test_size=TEST_SIZE)
        new_model = trainer.train(x_train, y_train, x_valid, y_valid)
        if model_cfg.model_name in ['HAN', 'MHAN', 'Self_Att']:
            att_res = new_model.get_attentions(SHOW_TEXT)
            visualize_attention(SHOW_TEXT, att_res, DATE, word2idx, model_cfg.model_name, N_LIMIT, LABEL)
    else:
        print(N_FOLDS, 'folds training')
        kf = StratifiedKFold(N_FOLDS, shuffle=True)
        kf.get_n_splits(data)
        output = {}
        for k, (train_index, valid_index) in enumerate(kf.split(data, labels)):
            print('Fold', k+1)
            model = HAN(model_cfg, embedding_matrix)
            trainer = Trainer(model, train_cfg, True)
            x_train, x_valid = data[train_index], data[valid_index]
            y_train, y_valid= to_categorical(labels[train_index]), to_categorical(labels[valid_index])
            new_model = trainer.train(x_train, y_train, x_valid, y_valid)
            predicted = new_model.predict(x_valid)
            y_true = np.argmax(y_valid, axis=1)
            y_pred = np.argmax(predicted, axis=1)
            res = list(np.array(precision_recall_fscore_support(y_true, y_pred)[0:3]).T.ravel())
            macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
            print('Fold-{:1d}'.format(k+1), 'macro_f1-{:2.4f}'.format(macro_f1))
            print('=================')
            res.append(macro_f1)
            output[k] = res
        print('average macro_f1:{:2.2f}'.format(np.mean([v[-1] for _,v in output.items()])*100))
        res_df = pd.DataFrame.from_dict(output, orient='index')
        res_df.columns = ['pre_0', 'recall_0', 'f1_0', 'pre_1', 'recall_1', 'f1_1', 'macro_f1']
        res_df.to_excel('./result/res.xlsx', index=None)

if __name__ == '__main__':
    training(data, labels, model_file, model_cfg, train_cfg, sys.argv[1])