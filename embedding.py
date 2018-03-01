# -*- coding: utf-8 -*-
'''使用gensim训练词向量'''

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from flashtext import KeywordProcessor
from utils import save_txt_data, read_line_data, clean_text
from utils import segment_sent, check_segment, strQ2B
from utils import fix_segment
#from wordsegment import load
from datetime import datetime
import re

'''
all_text = read_line_data('./data/raw_text.txt')

# 去重
distinct_text = list(set(all_text))
# 全角转半角
banjiao_text = [strQ2B(sent) for sent in distinct_text]

# 清理文本
cleaned_text = clean_text(banjiao_text)
save_txt_data('./data/cleaned_text.txt', cleaned_text, 'text')
print('Texts have been cleaned!')

# 分词
#load()
segmented_text = segment_sent(cleaned_text)
save_txt_data('./data/segmented_text.txt', segmented_text, 'text')
print('Sentences have been segmented!')
'''
# 修正分词错误
segmented_text = read_line_data('./data/bigger_corpus.txt')
segmented_text = fix_segment(segmented_text)
print('fix complete...')   

# 查看分词情况
#segmented_text = list(set(segmented_text))
#result, vocb = check_segment(segmented_text)
#t = {k:v for k,v in vocb.items() if k.startswith('发现')}

# 训练word2vec词向量
# todo: 生僻词标记为unknown，glove词向量
def word2vec(text, fname, ndims, window_size, min_cnt = 1):
    model = Word2Vec(text, min_count = min_cnt, window = window_size, size = ndims)
    # print(model.most_similar('苹果', topn = 5))
    word_vectors = model.wv
    word_vectors.save_word2vec_format(fname, binary = False)
    return model

#　可视化词向量
def visual_embeddings(word_list, embeddings):
    tsne = TSNE(perplexity = 30, n_components = 2, 
                init='pca', n_iter = 5000, method='exact')
    final_embeddings = [embeddings[word] for word in word_list]
    low_dim_embs = tsne.fit_transform(final_embeddings)
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(word_list):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

# 载入词向量
def load_embeddings(fname):
    embeddings_index = {}
    f = open(fname, encoding = 'utf-8')
    for k,line in enumerate(f.readlines()):
        if k != 0:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index

TEXT_FORMAT = 'word'
n_dim = 256
min_cnt = 5

if TEXT_FORMAT == 'word':
    segment_text = [sent.split() for sent in segmented_text]
    window_size = 5
elif TEXT_FORMAT == 'char':
    segment_text = [re.sub(' ', '', sent) for sent in segmented_text]
    segment_text = [[char for char in sent] for sent in segment_text]
    window_size = 8
    
print('Start Trainning Word2Vec ' + TEXT_FORMAT+ ' model')
date = datetime.now().date().strftime('%Y%m%d')
filename = './data/embeddings/' + TEXT_FORMAT + '_vectors_%sd_'%(n_dim) + date + '_%s.txt'%(min_cnt)
model_name =  './data/embeddings/' + TEXT_FORMAT + '_model_%sd_'%(n_dim) + date + '_%s'%(min_cnt)
model = word2vec(segment_text, filename, n_dim, window_size, min_cnt)
model.save(model_name)
w2v_embeddings = load_embeddings('./data/embeddings/word_vectors_256d_20171225_5.txt')
#visual_embeddings(word_list, w2v_embeddings)
