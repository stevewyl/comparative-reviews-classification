import pandas as pd
import numpy as np
import itertools
import re
import io

def load_data_and_labels(filename, datasets_name, mode):
    sents = {}
    df = pd.read_excel(filename)
    hidden_index = df.index[df['H'] == 1].tolist()
    sents['comp'] = df[df['Yes/No'] == 1]['cleaned_reviews']
    sents['non'] = df[df['Yes/No'] == 0]['cleaned_reviews']
    sents['equal'] = df[(df['平比'] == 1) & (df['H'] == 0)]['cleaned_reviews']
    sents['notequal'] = df[(df['Yes/No'] == 1) & (df['H'] == 0) & (df['平比'] != 1)]['cleaned_reviews']
    sents['most_comp'] =  df[(df['差比'] == 3) & (df['H'] == 0)]['cleaned_reviews']
    sents['hidden'] = df[df['H'] == 1]['cleaned_reviews']
    sents['not_hidden'] = df[(df['Yes/No'] == 1) & (df['H'] == 0)]['cleaned_reviews']
    
    # 合并数据集
    dataset = [sents[name].tolist() for name in datasets_name]
    x = list(itertools.chain(*dataset))
    y = [k for k,name in enumerate(datasets_name) for _ in range(sents[name].shape[0])]
    if mode == 'char':
        x = [re.sub(' ', '', sent) for sent in x]
        x = [[char for char in sent] for sent in x]
        x = [' '.join(sent) for sent in x]

    print('data load succeed')
    
    return x, np.asarray(y), hidden_index


# 载入词向量矩阵
def load_embeddings(fname, vocab, n_dim):
    embeddings_index = {}
    with open(fname, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(vocab) + 1, n_dim))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# fasttext词向量官方加载方式
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

# batch生成器
def batch_iter(data, labels, batch_size, shuffle=True, preprocessor=None):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        '''为数据集生成batch迭代器'''
        data_size = len(data)
        while True:
            # 每个轮次均随机化数据
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                if preprocessor:
                    yield preprocessor.transform(X, y)
                else:
                    yield X, y

    return num_batches_per_epoch, data_generator()
