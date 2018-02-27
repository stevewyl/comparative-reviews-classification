# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:07:36 2017

@author: steve
"""

'''平均词向量'''
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pprint import pprint
from collections import Counter


def load_data(data_file):
    df = pd.read_excel(data_file)
    comp = df[df['Yes/No'] == 1]['cleaned_reviews']
    non = df[df['Yes/No'] == 0]['cleaned_reviews']
    hidden = df[df['H'] == 1]['cleaned_reviews']
    not_hidden = df[(df['Yes/No'] == 1) & (df['H'] == 0)]['cleaned_reviews']
    return comp, non, hidden, not_hidden

# 载入词向量矩阵
def load_embeddings(fname, vocab, n_dim):
    embeddings_index = {}
    f = open(fname, encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((len(vocab) + 1, n_dim))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_x_y(dataset):
   x, y = [], []
   for i in range(len(dataset)):
      x = x + dataset[i].tolist()
      y += [i for _ in range(dataset[i].shape[0])]
   return x,y

def get_sequences(tokenizer, train_data, max_length):
    word_ids = tokenizer.texts_to_sequences(train_data)
    padded_seqs = pad_sequences(word_ids, maxlen = max_length)
    return padded_seqs
    
def eval_score(confusion_mat, num_labels):
    mat = np.array(confusion_mat)
    cnt_support = np.sum(confusion_mat, 1)
    cnt_total= np.sum(confusion_mat)
    res = {0:{}, 1:{}, 'total':{}}
    for idx in range(num_labels):
        precision = round(float(mat[idx][idx] / np.sum(mat.T[idx])), 4)
        recall = round(float(mat[idx][idx] / np.sum(mat[idx])), 4)
        res[idx]['pre'] = precision
        res[idx]['recall'] = recall
        res[idx]['f1']= round(float(2 * precision * recall / (precision + recall)), 4)
        res[idx]['ratio'] = round(float(cnt_support[idx] / cnt_total), 4)
    res['total']['pre'] = np.sum([res[idx]['pre'] * res[idx]['ratio'] for idx in range(num_labels)])
    res['total']['recall'] = np.sum([res[idx]['recall'] * res[idx]['ratio'] for idx in range(num_labels)])
    res['total']['f1'] = np.sum([res[idx]['f1'] * res[idx]['ratio'] for idx in range(num_labels)])
    res['total']['cnt'] = cnt_total
    return res    
    
def cv_train(x, y, model, n_folds = 10):
    kf = StratifiedKFold(n_folds, shuffle = True)
    kf.get_n_splits(x)
    comp_score, non_score, total_score = [], [], []
    for train_index, test_index in kf.split(x, y):
        X_train = [x[index] for index in train_index]
        X_test = [x[index] for index in test_index]
        y_train = [y[index] for index in train_index]
        y_test = [y[index] for index in test_index]
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        confusion_mat = confusion_matrix(y_test, predicted)
        eval_res = eval_score(confusion_mat, 2)
        comp_score.append([eval_res[0]['pre'], eval_res[0]['recall'], eval_res[0]['f1']])
        non_score.append([eval_res[1]['pre'], eval_res[1]['recall'], eval_res[1]['f1']])
        total_score.append(eval_res['total']['f1'])
    return comp_score, non_score, total_score
    
# 平均化词向量
# 有bug
def embedding_means(embedding_matrix, word2freq, sequences):
    res = []
    alpha = 0.0000000001
    for i in range(len(sequences)):
        arr = np.array([embedding_matrix[v]*word2freq[v] if v != 0 and v in word2freq else embedding_matrix[v] *alpha for v in sequences[i]])
        res.append(np.mean(arr, axis = 0))
    return np.array(res)
    
if __name__ == '__main__':
    comp, non, hidden, not_hidden = load_data('./data/jd_comp_final_v5.xlsx')
    EMBED_FILE = './data/embeddings/word_vectors_512d_20171225_1.txt'
    EMBED_DIMS = 512
    MAX_LEGTH = 100
    DATASET = [comp, non]
    x, y= get_x_y(DATASET)
    tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower = True, split = " ")
    tokenizer.fit_on_texts(x)
    vocab = tokenizer.word_index
    word2idx = {v:k for k,v in vocab.items()}
    all_text = [word for sent in x for word in sent.split()]
    word2freq = {vocab[k]:v/len(all_text) for k,v in Counter(all_text).items() if v>2 and k in vocab}

    embedding_matrix = load_embeddings(EMBED_FILE, vocab, EMBED_DIMS)
    X = embedding_means(embedding_matrix, word2freq,
                        get_sequences(tokenizer, x, MAX_LEGTH))

    model_svm = SGDClassifier(loss='hinge', penalty='l2', alpha = 0.001, random_state=2017)
    model_lr = LogisticRegression(penalty='l1')
    model_svc = SVC(C=2.0, kernel='rbf')
    model_nb = MultinomialNB(alpha = 0.1)
    
    score = cv_train(X, y, model_svm)
    pprint(score[0])
    print(np.mean([row[2] for row in score[0]]))