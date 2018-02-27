# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:11:48 2017

@author: steve
"""
from model_library import sent_word_HAN
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
from sklearn.preprocessing import Normalizer, LabelEncoder

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
    
    le_s = LabelEncoder().fit(df['sent_bin'])
    le_p = LabelEncoder().fit(df['product_cnt'])
    df['sentiment'] = le_s.transform(df['sent_bin'])
    df['prodcuts'] = le_p.transform(df['product_cnt'])
    hidden_index = df.index[df['H'] == 1].tolist()
    columns = ['cleaned_reviews', 'sentiment', 'prodcuts']
    comp = df[df['Yes/No'] == 1][columns]
    non = df[df['Yes/No'] == 0][columns]
    hidden = df[df['H'] == 1][columns]
    not_hidden = df[(df['Yes/No'] == 1) & (df['H'] == 0)][columns]
    print('text data load succeed')
    return comp, non, hidden, not_hidden, hidden_index



def get_x_y(dataset, mode):
    x, y, s, p= [], [], [], []
    for i in range(len(dataset)):
        x += dataset[i]['cleaned_reviews'].tolist()
        s += dataset[i]['sentiment'].tolist()
        y += [i for _ in range(dataset[i].shape[0])]
        p += dataset[i]['prodcuts'].tolist()
    if mode == 'char':
        x = [re.sub(' ','',sent) for sent in x]
        x = [[char for char in sent] for sent in x]
        x = [' '.join(sent) for sent in x]
    return x, y, s, p

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

def single_training(model, x, y, a, batch_size, n_epochs, test_size, num_labels,
                    max_words, text_mode, pre_trained):
    checkpoint = ModelCheckpoint(weights_name, monitor = 'val_model_output_acc', verbose = 1,
                                 save_best_only = True, mode = 'max')
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    history = model.fit({'text_inputs': x[0], 'addtional_inputs': a[0]},
                        {'model_output': y[0], 'att_output': y[0]},
                        batch_size = batch_size,
                        epochs = n_epochs,
                        validation_data = ([x[1],a[1]], [y[1],y[1]]),
                        callbacks = [early_stopping])
    model_yaml = model.to_yaml()
    with open(m_name, "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights(weights_name)
    predicted = model.predict([x[1],a[1]])[0]
    y_true = np.argmax(y[1], axis = 1)
    y_pred = np.argmax(predicted, axis = 1)
    print("\naccuracy score: {:.3f}".format(accuracy_score(y_true, y_pred)))
    print("\nconfusion matrix\n")
    print(confusion_matrix(y_true, y_pred))
    print("\nclassification report\n")
    print(classification_report(y_true, y_pred))
    # plot_loss_accuray(history)
    return y_true, y_pred, predicted

def model_build(add_shape, num_labels, max_words, PRE_TRAINED):
    sent_model, doc_model = sent_word_HAN(add_shape, max_words, MAX_SENTS, EMBED_DIMS, len(vocab),
                                          [384,256],[0.5,0.25,0.15], [0.3,0.15], num_labels, 64,
                                          'sigmoid', 'binary_crossentropy', ACTIVATION, PRE_TRAINED,
                                          embedding_matrix)
    return sent_model, doc_model

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

def cv_train(add_shape, x, y, a, batch_size, n_epochs, num_labels, max_words, 
             text_mode, pre_trained, model_name, tokenizer, n_folds = 10):             
    kf = StratifiedKFold(n_folds, shuffle = True, random_state = 2017)
    #kf.get_n_splits(x)
    comp_score, non_score, total_score = [], [], []
    i = 1

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    for train_index, test_index in kf.split(x, y):
        print('Fold ' + str(i) + ':')
        x_train = [x[index] for index in train_index]
        x_test = [x[index] for index in test_index]
        a_train = to_categorical([a[index] for index in train_index], add_shape)
        a_test = to_categorical([a[index] for index in test_index], add_shape)
        y_train = to_categorical([y[index] for index in train_index])
        y_test = to_categorical([y[index] for index in test_index])
        x_train = np.array(get_sequences(tokenizer, x_train, text_mode, max_words))
        x_test = np.array(get_sequences(tokenizer, x_test, text_mode, max_words))

        _,model = model_build(add_shape, num_labels, max_words, pre_trained)
        model.fit({'text_inputs': x_train, 'addtional_inputs': a_train}, 
                  {'model_output': y_train, 'att_output': y_train},
                  batch_size = batch_size,
                  epochs = n_epochs,
                  validation_data = ([x_test, a_test], [y_test, y_test]),
                  callbacks = [early_stopping])
        predicted = model.predict([x_test, a_test])[0]
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

def train(CV, x, a, y, tokenizer, date):
    if not CV:
        # 模型初始化
        _,doc_MODEL = model_build(ADD_SHAPE, NUM_LABELS, MAX_WORDS, PRE_TRAINED)
        # 切分训练集和测试集
        x_train, x_test, Y_train, Y_test = train_test_split(x, y, 
                                                            test_size = TEST_SIZE, 
                                                            random_state = 2017)
        x_train, x_test, a_train, a_test = train_test_split(x, a, 
                                                            test_size = TEST_SIZE, 
                                                            random_state = 2017)
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)
        a_train = to_categorical(a_train, ADD_SHAPE)
        a_test = to_categorical(a_test, ADD_SHAPE)
        x_train = np.array(get_sequences(tokenizer, x_train, TEXT_FORMAT, MAX_WORDS))
        x_test = np.array(get_sequences(tokenizer, x_test, TEXT_FORMAT, MAX_WORDS))
 
        x = [x_train, x_test]
        y = [Y_train, Y_test]
        a = [a_train, a_test]
        # 开始训练
        y_true, y_pred, prob = single_training(doc_MODEL, x, y, a, BATCH_SIZE, N_EPOCHS,
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
            score_1, score_2, score_3 = cv_train(ADD_SHAPE, x, y, a, BATCH_SIZE, N_EPOCHS,
                                                 NUM_LABELS, MAX_WORDS, TEXT_FORMAT,
                                                 PRE_TRAINED, MODEL_NAME, tokenizer, N_FOLDS)
            all_res[i] = [score_1, score_2, score_3]
            print("--- %s seconds ---" % (time.time() - start_time))
            if FOLDS_EPOCHS > 1 and i < FOLDS_EPOCHS - 1:
                print('waiting 180 seconds...')
                time.sleep(180)
        return all_res

if __name__ == '__main__':
    comp, non, hidden, not_hidden, hidden_index = read_x_data('./data/jd_comp_final_v5.xlsx')
    DATASET = [not_hidden, non]
    MODEL_NAME = 'sent_word_HAN'
    CUT_MODE = 'simple'
    TEXT_FORMAT = 'seq'
    BATCH_SIZE = 64
    N_EPOCHS = 10
    TEST_SIZE = 0.2
    NUM_LABELS = len(DATASET)
    EMBED_FILE= './data/embeddings/word_vectors_512d_20171217_1.txt'
    PRE_TRAINED = True
    EMBED_DIMS = 512
    ACTIVATION = 'relu'
    CV = False
    N_FOLDS = 10
    FOLDS_EPOCHS = 1
    
    x, y, s, p= get_x_y(DATASET, 'word')
    addtional_info = {'sent':s, 'product':p}
    
    tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                            lower = True, split = " ")
    tokenizer.fit_on_texts(x)
    vocab = tokenizer.word_index
    vocab['UNK'] = 0
    
    DATE = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
    m_name = './model/' + MODEL_NAME + '_' + DATE + '.yaml'
    weights_name = './model/' + MODEL_NAME + '_weights_' + DATE + '.hdf5'

    MAX_WORDS = 15
    MAX_SENTS = 6
    
    x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in x]
    print('load embeddings...')
    embedding_matrix = load_embeddings(EMBED_FILE, vocab, EMBED_DIMS)
    
    addtional_features = 'sent'
    print('using addtional features: ', addtional_features)
    a = addtional_info[addtional_features]
    ADD_SHAPE = len(set(a))
    
    result = train(CV, x, a, y, tokenizer, DATE)
    
    

    