import pandas as pd
import numpy as np
from utils import SentenceGetter
from sklearn_crfsuite import CRF
from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

data = pd.read_csv("./data/ner_dataset.csv", encoding="utf8")
data = data.fillna(method="ffill")
sentences = SentenceGetter(data).sentences

# 特征：
# 1. 词
# 2. 词性
# 3. 比较候选词（0or1）
# 4. 启发式位置（0or1）
# 5. 浅层句法

# 模板:
# 1. 该词前后三个词的所有特征（共5*7=35)
# 2. 该词的每个特征两两组合（共7个)
# 3. 相邻两个词的同一特征组合（共10*3=30)
# 4. 中心词词窗为1的同一特征组合（共4*3=12）
# 总计特征：84个






# 是否位于介词与比较词之间
def judge_location_1(sent, i):
    if i != 0 and i < len(sent)-1:
        if (sent[i-1][3] and sent[i+1][1] == 'p') or (sent[i+1][3] and sent[i-1][1] == 'p'):
            return 1
        else:
            return 0 
    else:
        return 0 

def judge_location_2(sent, i):
    if i != 0 and i < len(sent)-1:
        if (sent[i-1][3] and sent[i+1][1] == 's') or (sent[i+1][3] and sent[i-1][1] == 's'):
            return 1
        else:
            return 0 
    else:
        return 0 


def word2features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    dp = sent[i][2]
    comp = sent[i][3]
    loc_1 = judge_location_1(sent,i)
    loc_2 = judge_location_2(sent,i)

    features = {
        'word': word,
        'pos': pos,
        'comp': comp,
        'loc_1': loc_1,
        'loc_2': loc_2,
        'dp': dp,
        'word_pos': word + '_' + pos,
        'pos_comp': pos + '_' + str(comp),
        'pos_loc_1': pos + '_' + str(loc_1),
        'pos_loc_2': pos + '_' + str(loc_2),
        'pos_dp': pos + '_' + dp,
        'comp_loc_1': str(comp) + '_' + str(loc_1),
        'comp_loc_2': str(comp) + '_' + str(loc_2),
        'comp_dp': str(comp) + '_' + dp,
        'loc_1_dp': str(loc_1) + '_' + dp,
        'loc_2_dp': str(loc_2) + '_' + dp
    }

    if i > 3 and i < len(sent)-4:
        features.update({
            'word_+3': sent[i+3][0]
            'word_+2': sent[i+2][0]
            'word_+1': sent[i+1][0]
            'word_-3': sent[i-1][0]
            'word_-2': sent[i-2][0]
            'word_-1': sent[i-3][0]
            'pos_+3': sent[i+3][1]
            'pos_+2': sent[i+2][1]
            'pos_+1': sent[i+1][1]
            'pos_-3': sent[i-1][1]
            'pos_-2': sent[i-2][1]
            'pos_-1': sent[i-3][1]
            'dp_+3': sent[i+3][2]
            'dp_+2': sent[i+2][2]
            'dp_+1': sent[i+1][2]
            'dp_-3': sent[i-1][2]
            'dp_-2': sent[i-2][2]
            'dp_-1': sent[i-3][2]
            'word_word_1': sent[i-1][0] + '_' + word
            'word_word_2': word + '_' + sent[i+1][0]
        })
    else:
