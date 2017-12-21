# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:11:48 2017

@author: steve
"""
from model_library import char_word_HAN
import pandas as pd
import numpy as np
import time
import re
from pprint import pprint
from collections import Counter
    
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import model_from_yaml

from model_library import AttLayer
from utils import plot_loss_accuray, save_txt_data, split_sent

def read_x_data(filename):
    df = pd.read_excel(filename)
    hidden_index = df.index[df['H'] == 1].tolist()
    comp = df[df['Yes/No'] == 1]['cleaned_reviews']
    non = df[df['Yes/No'] == 0]['cleaned_reviews']
    hidden = df[df['H'] == 1]['cleaned_reviews']
    not_hidden = df[(df['Yes/No'] == 1) & (df['H'] == 0)]['cleaned_reviews']
    print('text data load succeed')
    return comp, non, hidden, not_hidden, hidden_index

def get_x_y(dataset):
    x, y = [], []
    for i in range(len(dataset)):
        x += dataset[i].tolist()
        y += [i for _ in range(dataset[i].shape[0])]
    x_c = [re.sub(' ','',sent) for sent in x]
    x_c = [[char for char in sent] for sent in x_c]
    x_c = [' '.join(sent) for sent in x_c]
    return x, x_c, y

def get_sequences(tokenizer, train_data, mode, max_length):
    if mode == 'seq':
        res = []
        for sent in train_data:
            word_ids = tokenizer.texts_to_sequences(sent)
            padded_seqs = pad_sequences(word_ids, maxlen = max_length)
            res.append(padded_seqs)
        return res
    elif mode == 'text':
        word_ids = tokenizer.texts_to_sequences(train_data)
        padded_seqs = pad_sequences(word_ids, maxlen = max_length)
        return padded_seqs
    
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

def single_training(model, x, y, batch_size, n_epochs, test_size, num_labels,
                    max_words, text_mode, pre_trained):
    checkpoint = ModelCheckpoint(weights_name, monitor = 'val_acc', verbose = 1,
                                 save_best_only = True, mode = 'max')
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    history = model.fit({'word_inputs': x[0][0], 'char_inputs': x[0][1]},
                        y[0],
                        batch_size = batch_size,
                        epochs = n_epochs,
                        validation_data = ([x[1][0],x[1][1]], y[1]),
                        callbacks = [checkpoint, early_stopping])
    model_yaml = model.to_yaml()
    with open(m_name, "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights(weights_name)
    predicted = model.predict([x[1][0],x[1][1]])
    y_true = np.argmax(y[1], axis = 1)
    y_pred = np.argmax(predicted, axis = 1)
    print("\naccuracy score: {:.3f}".format(accuracy_score(y_true, y_pred)))
    print("\nconfusion matrix\n")
    print(confusion_matrix(y_true, y_pred))
    print("\nclassification report\n")
    print(classification_report(y_true, y_pred))
    plot_loss_accuray(history)
    return y_true, y_pred, predicted

def model_build(num_labels, max_words, PRE_TRAINED):
    model = char_word_HAN(max_words, MAX_SENTS, EMBED_DIMS, [len(vocab_w), len(vocab_c)],
                          [256,128],[0.4,0.25,0.15], [0.25,0.15], num_labels, 64,
                          'sigmoid', 'binary_crossentropy', ACTIVATION, PRE_TRAINED,
                          [embed_mat_w, embed_mat_c])
    return model

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


def cv_train(x, y, batch_size, n_epochs, num_labels, max_words, 
             text_mode, pre_trained, model_name, tokenizer, n_folds = 10):             
    kf = StratifiedKFold(n_folds, shuffle = True)
    kf.get_n_splits(x[0])
    comp_score, non_score, total_score = [], [], []
    i = 1
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    for train_index, test_index in kf.split(x[0], y):
        print('Fold ' + str(i) + ':')
        x_train_w = [x[0][index] for index in train_index]
        x_test_w = [x[0][index] for index in test_index]
        x_train_c = [x[1][index] for index in train_index]
        x_test_c = [x[1][index] for index in test_index]
        y_train = to_categorical([y[index] for index in train_index])
        y_test = to_categorical([y[index] for index in test_index])
        x_train_w = np.array(get_sequences(tokenizer[0], x_train_w, text_mode, max_words[0]))
        x_test_w = np.array(get_sequences(tokenizer[0], x_test_w, text_mode, max_words[0]))
        x_train_c = np.array(get_sequences(tokenizer[1], x_train_c, text_mode, max_words[1]))
        x_test_c = np.array(get_sequences(tokenizer[1], x_test_c, text_mode, max_words[1]))

        model = model_build(num_labels, max_words, pre_trained)
        model.fit({'word_inputs': x_train_w, 'char_inputs': x_train_c}, 
                  y_train,
                  batch_size = batch_size,
                  epochs = n_epochs,
                  validation_data = ([x_test_w, x_test_c], y_test),
                  callbacks = [early_stopping])
        predicted = model.predict([x_test_w, x_test_c])
        y_true = np.argmax(y_test, axis = 1)
        y_pred = np.argmax(predicted, axis = 1)
        confusion_mat = confusion_matrix(y_true, y_pred)
        eval_res = eval_score(confusion_mat, num_labels)
        comp_score.append([eval_res[0]['pre'], eval_res[0]['recall'], eval_res[0]['f1']])
        non_score.append([eval_res[1]['pre'], eval_res[1]['recall'], eval_res[1]['f1']])
        total_score.append(eval_res['total']['f1'])
        print('comp average f1 score: ', np.mean([s[2] for s in comp_score]))
        print('non average f1 score: ', np.mean([s[2] for s in non_score]))
        print('===================')
        i += 1
    print('total average f1 score: ', np.mean(total_score))
    return comp_score, non_score, total_score

def train(CV, x, y, tokenizer, date):
    if not CV:
        # 模型初始化
        doc_MODEL = model_build(NUM_LABELS, MAX_WORDS, PRE_TRAINED)
        # 切分训练集和测试集
        x_train_w, x_test_w, Y_train, Y_test = train_test_split(x[0], y, 
                                                            test_size = TEST_SIZE, 
                                                            random_state = 2017)
        x_train_c, x_test_c, Y_train, Y_test = train_test_split(x[1], y, 
                                                    test_size = TEST_SIZE, 
                                                    random_state = 2017)
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)
        x_train_w = np.array(get_sequences(tokenizer[0], x_train_w, TEXT_FORMAT, MAX_WORDS[0]))
        x_test_w = np.array(get_sequences(tokenizer[0], x_test_w, TEXT_FORMAT, MAX_WORDS[0]))
        x_train_c = np.array(get_sequences(tokenizer[1], x_train_c, TEXT_FORMAT, MAX_WORDS[1]))
        x_test_c = np.array(get_sequences(tokenizer[1], x_test_c, TEXT_FORMAT, MAX_WORDS[1]))

        x = [[x_train_w, x_train_c], [x_test_w, x_test_c]]
        y = [Y_train, Y_test]
        # 开始训练
        y_true, y_pred, prob = single_training(doc_MODEL, x, y, BATCH_SIZE, N_EPOCHS,
                                               TEST_SIZE, NUM_LABELS, MAX_WORDS, 
                                               TEXT_FORMAT, PRE_TRAINED)
        # 可视化        
        #print('plotting roc curve...')
        #plot_roc_curve(Y_test, prob, NUM_LABELS, 1)
        
        # best_model = model_predict('','', x_test[0:20])
        # return [w_r_idx, sent_all_att_0, sent_att_0, doc_att_0]
        return y_true, y_pred, prob

    else:
        all_res = {}
        for i in range(FOLDS_EPOCHS):
            start_time = time.time()
            print('No.' + str(i+1) + ' ' + str(N_FOLDS) + ' folds training starts...')
            score_1, score_2, score_3 = cv_train(x, y, BATCH_SIZE, N_EPOCHS,
                                                 NUM_LABELS, MAX_WORDS, TEXT_FORMAT,
                                                 PRE_TRAINED, MODEL_NAME, tokenizer, N_FOLDS)
            all_res[i] = [score_1, score_2, score_3]
            print("--- %s seconds ---" % (time.time() - start_time))
            if FOLDS_EPOCHS > 1:
                print('waiting 180 seconds...')
                time.sleep(180)
        return all_res

if __name__ == '__main__':
    comp, non, hidden, not_hidden, hidden_index = read_x_data('./data/jd_comp_final_v3.xlsx')
    DATASET = [not_hidden, non]
    MODEL_NAME = 'char_word_HAN'
    CUT_MODE = 'simple'
    TEXT_FORMAT = 'seq'
    BATCH_SIZE = 64
    N_EPOCHS = 3
    TEST_SIZE = 0.1
    NUM_LABELS = len(DATASET)
    EMBED_FILE_word = './data/embeddings/word_vectors_512d_20171217_1.txt'
    EMBED_FILE_char = './data/embeddings/char_vectors_512d_20171219_1.txt'
    PRE_TRAINED = True
    EMBED_DIMS = 512
    ACTIVATION = 'selu'
    CV = False
    N_FOLDS = 10
    FOLDS_EPOCHS = 1
    
    x_w, x_c, y = get_x_y(DATASET)
    
    tokenizer_w = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                            lower = True, split = " ")
    tokenizer_c = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                            lower = True, split = " ")
    tokenizer_w.fit_on_texts(x_w)
    tokenizer_c.fit_on_texts(x_c)
    vocab_w = tokenizer_w.word_index
    vocab_w['UNK'] = 0
    vocab_c = tokenizer_c.word_index
    vocab_c['UNK'] = 0
    
    DATE = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
    m_name = './model/' + MODEL_NAME + '_' + DATE + '.yaml'
    weights_name = './model/' + MODEL_NAME + '_weights_' + DATE + '.hdf5'

    MAX_WORDS = [20,30]
    MAX_SENTS = [6,6]
    
    x_w = [split_sent(sent, MAX_WORDS[0], MAX_SENTS[0], CUT_MODE) for sent in x_w]
    x_c = [split_sent(sent, MAX_WORDS[1], MAX_SENTS[1], CUT_MODE) for sent in x_c]
    
    embed_mat_w = load_embeddings(EMBED_FILE_word, vocab_w, EMBED_DIMS)
    embed_mat_c = load_embeddings(EMBED_FILE_char, vocab_c, EMBED_DIMS)
    
    result = train(CV, [x_w, x_c], y, [tokenizer_w, tokenizer_c], DATE)
    
    

    