import re
import numpy as np
import numpy.random as nprnd
import pandas as pd
from datetime import datetime
import time
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import Normalizer

from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import read_line_data, split_sent, plot_loss_accuray
from utils import customed_heatmap
from model_library import  HAN
from model_library import get_attention

def word2vec(text, fname, ndims, window_size, min_cnt = 1):
    model = Word2Vec(text, min_count = min_cnt, window = window_size, size = ndims)
    word_vectors = model.wv
    word_vectors.save_word2vec_format(fname, binary = False)
    return model

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

def load_data(fname):
    df = pd.read_csv('./data/' + fname, header = None)
    return df[0].tolist()

def get_char_text(fname, training = False):
    if training:
        segmented_text = read_line_data(fname)
        segmented_text = list(set(segmented_text))
    else:
        segmented_text = load_data(fname)
    without_sapce_text = [re.sub(' ', '', sent) for sent in segmented_text]
    char_text = [[char for char in sent] for sent in without_sapce_text]
    if training:
        return char_text
    else:
        return [' '.join(sent) for sent in char_text]

def get_x_y(dataset):
    x, y = [], []
    for i in range(len(dataset)):
        x += dataset[i]
        y += [i for _ in range(len(dataset[i]))]
    y = to_categorical(y)
    return x, y

def get_sequences(train_data, max_length):
    res = []
    for sent in train_data:
        word_ids = tokenizer.texts_to_sequences(sent)
        padded_seqs = pad_sequences(word_ids, maxlen = max_length)
        res.append(padded_seqs)
    return res

def visualize_attention(x_test, reviews_length, word2idx, label):
    x_samples = np.array([x_test[k] for k,v in enumerate(y_true) if v == label])
    #x_length = np.array([reviews_length[k] for k,v in enumerate(y_true) if v == label])
    #x_1_samples = np.array([x_test[k] for k,v in enumerate(y_true) if v == 1])
    random_index = nprnd.randint(x_samples.shape[0], size = SHOW_SAMPLES_CNT)
    select_samples = x_samples[random_index]
    #select_samples_len = x_length[random_index]
    sent_all_att, sent_att, doc_att, word_idx = get_attention(sent_MODEL, 
                                                              doc_MODEL, 
                                                              select_samples)
    #sent_all_att = [sent_all_att[i][0:select_samples_len[i]] for i in range(SHOW_SAMPLES_CNT)]
    text_sent = [[word2idx[idx] for sub in select_samples[i] for idx in sub] for i in range(SHOW_SAMPLES_CNT)]
    normalizer = Normalizer()
    all_att = normalizer.fit_transform(sent_all_att)
    customed_heatmap(all_att, text_sent, N_LIMIT)

    important_words = [[word2idx[idx] for idx in word_idx[w_idx]] 
                        for w_idx in range(SHOW_SAMPLES_CNT)]
    print('some important keywords:')
    pprint(important_words)
    return sent_all_att, sent_att, doc_att

if __name__ == '__main__':
    comp = get_char_text('comp_reviews_word_03.csv')
    non = get_char_text('non_comp_reviews_word_03.csv')
    hidden = get_char_text('hidden_reviews_word_03.csv')
    not_hidden = get_char_text('not_hidden_reviews_word_03.csv')

    EMBED_TRAIN = False
    SHOW_SAMPLES_CNT = 15
    N_DIMS = 256
    TEST_SIZE = 0.2
    MAX_WORDS = 25
    MAX_SENTS = 8
    N_LIMIT = MAX_WORDS * MAX_SENTS
    DATASET = [not_hidden, non]
    NUM_LABELS = len(DATASET)
    BATCH_SIZE = 64
    N_EPOCHS = 4
    DATE = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
    CUT_MODE = 'simple'
    classifier = 'sigmoid'
    loss_function = 'binary_crossentropy'

    date = datetime.now().date().strftime('%Y%m%d')
    weights_name = './model/HAN_char_weights_' + DATE + '.hdf5'

    x, y = get_x_y(DATASET)
    x_copy = x
    reviews_length = np.array([len(sent.split()) for sent in x_copy])
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower = True, split = " ")
    tokenizer.fit_on_texts(x)
    vocab = tokenizer.word_index
    vocab['UNK'] = 0
    word2idx = {v:k for k,v in vocab.items()}
    
    if EMBED_TRAIN:
        print('start training char embeddings...')
        embedding_fname = './data/char_vectors_%sd_'%(N_DIMS) + date + '.txt'
        char_text = get_char_text('./data/baidu_segment_text.txt', True)
        embedd_model = word2vec(char_text, embedding_fname, N_DIMS, 5, 5)
    embeding_fname = './data/char_vectors_256d_20171206.txt'
    embedding_matrix = load_embeddings(embeding_fname, vocab, N_DIMS)
    
    x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in x]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, 
                                                    test_size = TEST_SIZE, 
                                                    random_state = 2017)
    X_train = np.array(get_sequences(X_train, MAX_WORDS))
    X_test = np.array(get_sequences(X_test, MAX_WORDS))

    print('building HAN Model...')
    sent_MODEL, doc_MODEL = HAN(MAX_WORDS, MAX_SENTS, N_DIMS, len(vocab), [256,128],
                                [0.4,0.3,0.1], 0.25, NUM_LABELS, 64, 
                                classifier, loss_function,
                                'relu', True, embedding_matrix)
    
    checkpoint = ModelCheckpoint(weights_name, monitor = 'val_acc', verbose = 1,
                                 save_best_only = True, mode = 'max')
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    print('start training...')
    history = doc_MODEL.fit(X_train, Y_train,
                        batch_size = BATCH_SIZE,
                        epochs = N_EPOCHS,
                        validation_data = (X_test, Y_test),
                        callbacks = [checkpoint, early_stopping])
    predicted = doc_MODEL.predict(X_test)
    y_true = np.argmax(Y_test, axis = 1)
    y_pred = np.argmax(predicted, axis = 1)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    plot_loss_accuray(history)
    
    sent_all_att_0, sent_att_0, doc_att_0 = visualize_attention(X_test, reviews_length, word2idx, 0)

    
