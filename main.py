import re
import time
from reader import load_data_and_labels
from trainer import Trainer
from model_library import TextCNNBN, TextInception, convRNN, HAN, MHAN, SelfAtt, fasttext, Bi_RNN
from config import ModelConfig, TrainingConfig
from utils import customed_heatmap, read_line_data

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

config_file = 'config.json'
weight_file = 'model_weights.h5'

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
EMBED_FILE = './data/embeddings/word_vectors_256d_20171228_5.txt'
if EMBED_FILE:
    EMBED_TYPE = re.findall(r'(?<=/)\w+(?=_v)',EMBED_FILE)[0]
    EMBED_DIMS = int(re.findall(r'(?<=_)\d+(?=d)',EMBED_FILE)[0])
else:
    EMBED_TYPE = 'scratch'
    EMBED_DIMS = 256

# 读入数据
sent, labels, _ = load_data_and_labels('./data/jd_comp_final_v5.xlsx', ['not_hiden', 'non'], 'word')
if PREDICT:
    df_predict = pd.read_excel('./data/jd_20w_v2.xlsx')
    predict_text = df_predict['segment'].tolist()
    sent = sent + predict_text

# 初始化文本->index
tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower = True, split = " ")
tokenizer.fit_on_texts(sent)
vocab = tokenizer.word_index
vocab['UNK'] = 0
word2idx = {v:k for k,v in vocab.items()}

# 模型及权重保存路径
DATE = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
model_file = './model/' + MODEL_NAME + '_' + DATE + '.yaml'
weight_file = './model/' + MODEL_NAME + '_weights_' + DATE + '.hdf5'

# HAN模型需要的文本输入格式
if MODEL_NAME in ['HAN','HMAN']:
    print('prepare inputs for HAN series model...')
    if EMBED_TYPE == 'word' or 'scratch':
        MAX_WORDS = 20
        MAX_SENTS = 5
    elif EMBED_TYPE == 'char':
        MAX_WORDS = 30
        MAX_SENTS = 6
    N_LIMIT = MAX_WORDS * MAX_SENTS
    x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in x]
    if PREDICT:
        p_x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in predict_text]
    TEXT_FORMAT = 'seq'
    new_name = MODEL_NAME + '_' + str(MAX_WORDS) + '_' + str(MAX_SENTS)
    m_name = './model/' + new_name + '_' + DATE + '.yaml'
    weights_name = './model/' + new_name + '_weights_' + DATE + '.hdf5'

# 初始化参数设置
model_cfg = ModelConfig(MAX_WORDS, MAX_SENTS, EMBED_DIMS, len(vocab), MODEL_NAME, ntags=2)
train_cfg = TrainingConfig()