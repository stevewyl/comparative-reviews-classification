# to-do:
# 1. 误差分析(ongoing)
# 2. 词向量细化(current using gensim word2vec model)
# 3. 句子切分，HAN模型(V2 ok 0.78-0.8)
# 4. 可视化attention模型（heatmap已完成，细节优化 ongoing）（***）
# 5. conv-RNN模型(v1 ok 0.74-0.76)
# 6. 交叉验证(ok) ---> 统计经常被误分的样本（基本结束，比较不同词向量维度）
# 7. 权重初始化方式(relu:He, tanh:Xaiver, ok)
# 8. 参数单独新建一个文件(ini后缀) ---> configparser.ConfigParser()
# 9. roc曲线（ongoing，有bug)

# bug:
# 1. 部分模型不能保存 'rawunicodeescape' codec 错误（保存为yaml文件，已解决）


import pandas as pd
import numpy as np
import time
import re
from collections import Counter
from pprint import pprint
import numpy.random as nprnd
import math

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_yaml

from model_library import TextCNNBN, TextInception, convRNN, HAN, HMAN, SelfAtt
from model_library import AttLayer, fasttext, one_hot_mdoel, bi_rnn, TextCNN
from tools import get_attention
from utils import plot_loss_accuray, save_txt_data, split_sent
from utils import customed_heatmap, read_line_data

def read_x_data(df):
    le = LabelEncoder().fit(df['sent_bin'])
    df['sentiment'] = le.transform(df['sent_bin'])
    hidden_index = df.index[df['H'] == 1].tolist()
    comp = df[df['Yes/No'] == 1][['cleaned_reviews', 'sentiment']]
    non = df[df['Yes/No'] == 0][['cleaned_reviews', 'sentiment']]
    equal = df[(df['平比'] == 1) & (df['H'] == 0)][['cleaned_reviews', 'sentiment']]
    notequal = df[(df['Yes/No'] == 1) & (df['H'] == 0) & (df['平比'] != 1)][['cleaned_reviews', 'sentiment']]
    most_comp =  df[(df['差比'] == 3) & (df['H'] == 0)][['cleaned_reviews', 'sentiment']]
    hidden = df[df['H'] == 1][['cleaned_reviews', 'sentiment']]
    not_hidden = df[(df['Yes/No'] == 1) & (df['H'] == 0)][['cleaned_reviews', 'sentiment']]
    print('text data load succeed')
    return comp, non, hidden, not_hidden, equal, notequal, most_comp, hidden_index

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

# get the text and target
def get_x_y(dataset, mode):
    x, y, s= [], [], []
    for i in range(len(dataset)):
        x += dataset[i]['cleaned_reviews'].tolist()
        s += dataset[i]['sentiment'].tolist()
        y += [i for _ in range(dataset[i].shape[0])]
    if mode == 'char':
        x = [re.sub(' ','',sent) for sent in x]
        x = [[char for char in sent] for sent in x]
        x = [' '.join(sent) for sent in x]
    return x, y, s

# 对文本进行序列化处理
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

# 根据混淆矩阵计算f1分数
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

def remove_seq(text):
    return [re.sub('UNK', '', ' '.join(sent)) for sent in text]


# 误差分析（句子长度，术语出现次数，比较特征词出现次数，是否是隐性比较句等）
def error_analysis(x_test, y_true, y_pred, prob, date, text_mode, train_mode):
    close_samples_idx = [i for i,sample in enumerate(prob) if np.abs(sample[0]-sample[1])<0.25]
    totally_wrong_idx = [i for i,s in enumerate(prob) if np.abs(s[0]-s[1])>0.9 and y_true[i] < y_pred[i]]
    close_samples = [x_test[i] for i in close_samples_idx]
    wrong_samples = [x_test[i] for i in totally_wrong_idx]
    diff_recall_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i] and y_pred[i] != 0]
    diff_precision_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i] and y_true[i] != 0]
    mutual_idx_1 = list(set(diff_recall_idx).intersection(set(close_samples_idx)))
    mutual_idx_2 = list(set(diff_precision_idx).intersection(set(close_samples_idx)))
    mutual_idx = list(set(mutual_idx_1 + mutual_idx_2))
    diff_recall = [x_test[i] for i in diff_recall_idx]
    diff_precision = [x_test[i] for i in diff_precision_idx]
    close_wrong_samples = [x_test[i] for i in mutual_idx]
    
    if text_mode == 'seq':
        diff_precision = remove_seq(diff_precision)
        diff_recall = remove_seq(diff_recall)
        close_samples = remove_seq(close_samples)
        close_wrong_samples = remove_seq(close_wrong_samples)
        wrong_samples = remove_seq(wrong_samples)

    if train_mode == 'single':
        date_model = date + '_' + MODEL_NAME
        save_txt_data('./result/wrong_samples_precison_' + date_model + '.txt',
                      diff_precision, text_mode)
        save_txt_data('./result/wrong_samples_recall_' + date_model + '.txt',
                      diff_recall, text_mode)
        save_txt_data('./result/close_samples_' + date_model + '.txt',
                      close_samples, text_mode)
        save_txt_data('./result/mutual_wrong_samples_' + date_model + '.txt',
                      close_wrong_samples, text_mode)
        save_txt_data('./result/totally_wrong_samples_' + date_model + '.txt',
                      wrong_samples, text_mode)
        
    return diff_precision, diff_recall, wrong_samples

# 查看不同类别的文本的错误比例
def check_hidden(df, wrong_samples):
    wrong_samples = [re.sub(' ','',sent) for sent in wrong_samples]
    hidden_samples = df['cleaned_reviews'].tolist()
    hidden_samples = [re.sub(' ','',sent) for sent in hidden_samples]
    cnt = 0
    for s in wrong_samples:
        if s in hidden_samples:
            cnt += 1
    return round(cnt / len(wrong_samples), 4)

# 各个模型的结构初始化及模型超参设置
def model_build(name, num_labels, max_words, pre_trained, plot_structure = False):
    if num_labels == 2:
        classifier = 'sigmoid'
        loss_function = 'binary_crossentropy'
    else:
        classifier = 'softmax'
        loss_function = 'categorical_crossentropy'

    if name == 'convRNN':
        model = convRNN(max_words, EMBED_DIMS, len(vocab), 256, 
                        0.25, [384,256], 1, 'same', 5, 128, 
                        ACTIVATION, num_labels, classifier,
                        loss_function, pre_trained, embedding_matrix)
    elif name == 'TextCNNBN':
        model = TextCNNBN(max_words, EMBED_DIMS, len(vocab), 
                          [3,4,5], [256,128], 4, 'same', 256,
                          num_labels, ACTIVATION, classifier,
                          loss_function, pre_trained, embedding_matrix)
    elif name == 'Inception':
        model = TextInception(max_words, EMBED_DIMS, len(vocab),
                              [[1],[1,3],[3,5],[3]], [256,128], 
                              'same', 0.5, 256, num_labels, 
                              ACTIVATION, classifier,loss_function, 
                              pre_trained, embedding_matrix)
    elif name == 'HAN':
        model = HAN(max_words, MAX_SENTS, EMBED_DIMS, len(vocab), 
                    [256,128], [0.4,0.25,0.15], [0.25,0.15], 
                    num_labels, 64, classifier, loss_function,
                    ACTIVATION, pre_trained, embedding_matrix)
    elif name == 'HMAN':
        model = HMAN([4,2], max_words, MAX_SENTS, EMBED_DIMS, len(vocab), 
                     [256,128], [0.4,0.25,0.15], [0.25,0.15], 
                     num_labels, 64, classifier, loss_function,
                    ACTIVATION, pre_trained, embedding_matrix)
    elif name == 'SelfAtt':
        model = SelfAtt(10, max_words, EMBED_DIMS, len(vocab), 256, 
                        0.25, 0.15, num_labels, 128, classifier, 
                        loss_function, ACTIVATION, pre_trained,
                        embedding_matrix)
    elif name == 'fasttext':
        model = fasttext(len(vocab), max_words, EMBED_DIMS, embedding_matrix,
                         pre_trained, num_labels, classifier, loss_function)
    elif name == 'one-hot':
        model = one_hot_mdoel(len(vocab), 0.5, 512, num_labels, ACTIVATION,
                              classifier, loss_function)
    elif name == 'bi_rnn':
        model = bi_rnn(len(vocab), EMBED_DIMS, max_words, embedding_matrix,
                       pre_trained, 256, [0.25,0.15], num_labels, classifier,
                       loss_function)
    elif name == 'TextCNN':
        model = TextCNN(len(vocab), EMBED_DIMS, max_words, embedding_matrix,
                        pre_trained, 256, [3,4,5], 1, [5,5,5], 0.5, num_labels,
                        classifier, loss_function, 256)
    else:
        print('This model does not exist in model_library.py')

    # plot the model structure
    if plot_structure:
        plot_model(model, to_file = './result/model_' + name + '.png')

    return model

# 加载已经保存好的模型和权重矩阵
def load_model(model_file, weights_file):
    yaml_file = open(model_file, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    custom_layers = {'AttLayer': AttLayer}
    loaded_model = model_from_yaml(loaded_model_yaml, 
                                   custom_objects = custom_layers)
    loaded_model.load_weights(weights_file)
    return loaded_model

# 学习率衰减策略
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch) / epochs_drop))
    return lrate

# 10折交叉验证
def cv_train(x, y, batch_size, n_epochs, num_labels, max_words, 
             text_mode, pre_trained, model_name, tokenizer, n_folds = 10):             
    kf = StratifiedKFold(n_folds, shuffle = True)
    kf.get_n_splits(x)
    comp_score, non_score, total_score = [], [], []
    wrong_samples = {'pre':[], 'rec':[], 'mutal':[]}
    i = 1
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    lrate = LearningRateScheduler(step_decay)

    for train_index, test_index in kf.split(x, y):
        print('Fold ' + str(i) + ':')
        X_train = [x[index] for index in train_index]
        X_test = [x[index] for index in test_index]
        y_train = to_categorical([y[index] for index in train_index])
        y_test = to_categorical([y[index] for index in test_index])
        x_train = np.array(get_sequences(tokenizer, X_train, text_mode, max_words))
        x_test = np.array(get_sequences(tokenizer, X_test, text_mode, max_words))
        if model_name in ['HAN','HMAN']:
            _, model = model_build(model_name, num_labels, max_words, pre_trained)
        else:
            model = model_build(model_name, num_labels, max_words, pre_trained)
        model.fit(x_train, y_train,
                  batch_size = batch_size,
                  epochs = n_epochs,
                  validation_data = (x_test, y_test),
                  callbacks = [early_stopping, lrate])
        predicted = model.predict(x_test)
        y_true = np.argmax(y_test, axis = 1)
        y_pred = np.argmax(predicted, axis = 1)

        diff_pre, diff_rec, m_samples = error_analysis(X_test, y_true, y_pred, 
                                                       predicted, DATE, text_mode, 
                                                       'cross_fold')
        wrong_samples['pre'].append(diff_pre)
        wrong_samples['rec'].append(diff_rec)
        wrong_samples['mutal'].append(m_samples)
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
    return comp_score, non_score, total_score, wrong_samples

# 单次模型
def single_training(model, x, y, batch_size, n_epochs, test_size, num_labels,
                    max_words, text_mode, pre_trained):
    tensorboard = TensorBoard(log_dir = './tmp/log',
                              histogram_freq = 0,
                              write_graph = True,
                              write_grads = True)
    checkpoint = ModelCheckpoint(weights_name, monitor = 'val_acc', verbose = 1,
                                 save_best_only = True, mode = 'max')
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    lrate = LearningRateScheduler(step_decay)

    history = model.fit(x[0], y[0],
                        batch_size = batch_size,
                        epochs = n_epochs,
                        validation_data = (x[1], y[1]),
                        callbacks = [checkpoint, early_stopping, lrate])
    if SAVE_MODEL:
        model_yaml = model.to_yaml()
        with open(m_name, "w") as yaml_file:
            yaml_file.write(model_yaml)
        #model.save_weights(weights_name)
    predicted = model.predict(x[1])
    y_true = np.argmax(y[1], axis = 1)
    y_pred = np.argmax(predicted, axis = 1)
    print("\naccuracy score: {:.3f}".format(accuracy_score(y_true, y_pred)))
    print("\nconfusion matrix\n")
    print(confusion_matrix(y_true, y_pred))
    print("\nclassification report\n")
    print(classification_report(y_true, y_pred))
    plot_loss_accuray(history)
    
    return y_true, y_pred, predicted, history, model

# 载入训练好的模型预测新样本
def model_predict(model_file, weigths_file, test_data):
    best_model = load_model(model_file, weigths_file)
    best_model.predict(test_data)
    
    return best_model
    
# 可视化attention权重
def visualize_attention(x_test, y_true, sent_model, doc_model, date, word2idx, label, rand):
    print('Label:', str(label))
    #x_samples = np.array([x_test[k] for k,v in enumerate(y_true) if v == label])
    if rand:
        random_index = nprnd.randint(x_samples.shape[0], size = SHOW_SAMPLES_CNT)
        select_samples = x_samples[random_index]
    else:
        # select_samples = x_samples[0:SHOW_SAMPLES_CNT]
        select_samples = x_test
    sent_all_att, doc_all_att = get_attention(sent_model, 
                                              doc_model, 
                                              select_samples,
                                              MODEL_NAME)
    text_sent = [[word2idx[idx] for sub in select_samples[i] for idx in sub] for i in range(5)]
    normalizer_sent = Normalizer()
    normalizer_doc = Normalizer()
    att_sent = normalizer_sent.fit_transform(sent_all_att)
    att_doc = normalizer_doc.fit_transform(doc_all_att)

    customed_heatmap(att_sent, text_sent, N_LIMIT, date, label, 'sent')
    customed_heatmap(att_doc[:,::-1].T, text_sent, N_LIMIT, date, label, 'doc')
    #important_words = [[word2idx[idx] for idx in word_idx[w_idx]] 
    #                    for w_idx in range(SHOW_SAMPLES_CNT)]
    #print('some important keywords:')
    #pprint(important_words)
    return sent_all_att, doc_all_att

def train(CV, x, y, tokenizer, date):
    if not CV:
        # 模型初始化
        if MODEL_NAME in ['HAN','HMAN']:
            sent_MODEL, doc_MODEL = model_build(MODEL_NAME, NUM_LABELS, MAX_WORDS, PRE_TRAINED)
        else:
            doc_MODEL = model_build(MODEL_NAME, NUM_LABELS, MAX_WORDS, PRE_TRAINED)
        # 切分训练集和测试集
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, 
                                                            test_size = TEST_SIZE) 
                                                            #random_state = 2017)
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)
        x_train = np.array(get_sequences(tokenizer, X_train, TEXT_FORMAT, MAX_WORDS))
        x_test = np.array(get_sequences(tokenizer, X_test, TEXT_FORMAT, MAX_WORDS))
        if MODEL_NAME == 'one-hot':
            x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
            x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
        x = [x_train, x_test]
        y = [Y_train, Y_test]
        # 开始训练
        y_true, y_pred, prob, history, model = single_training(doc_MODEL, x, y, BATCH_SIZE, N_EPOCHS,
                                               TEST_SIZE, NUM_LABELS, MAX_WORDS, 
                                               TEXT_FORMAT, PRE_TRAINED)
        
        print('error_analysis...')
        res = error_analysis(X_test, y_true, y_pred, prob, DATE, TEXT_FORMAT, 'single')
        
        if CHECK_HIDDEN:
            check_hidden(hidden, res[3][2])
        
        # 可视化
        if MODEL_NAME in ['HAN', 'HMAN'] and ATTENTION_V:
            print('attention weights visualization...')
            sent_all_att_0, doc_att_0 = visualize_attention(x_test, y_true, sent_MODEL, doc_MODEL, date, word2idx, 0, RAND)
            #sent_all_att_1, doc_att_1 = visualize_attention(x_test, y_true, sent_MODEL, doc_MODEL, date, word2idx, 1, RAND)

        #print('plotting roc curve...')
        #plot_roc_curve(Y_test, prob, NUM_LABELS, 1)
        
        # best_model = model_predict('','', x_test[0:20])
        # return [w_r_idx, sent_all_att_0, sent_att_0, doc_att_0]
        if MODEL_NAME in ['HAN','HMAN']:
            return  y_true, y_pred, prob, res, history, model, sent_MODEL
        else:
            return  y_true, y_pred, prob, res, history, model

    else:
        all_res = {}
        for i in range(FOLDS_EPOCHS):
            start_time = time.time()
            print('No.' + str(i+1) + ' ' + str(N_FOLDS) + ' folds training starts...')
            score_1, score_2, score_3, wrong_samples = cv_train(x, y, BATCH_SIZE, N_EPOCHS,
                                                       NUM_LABELS, MAX_WORDS, TEXT_FORMAT,
                                                       PRE_TRAINED, MODEL_NAME, tokenizer, N_FOLDS)
            all_res[i] = [score_1, score_2, score_3, wrong_samples]
            keys = list(wrong_samples.keys())
            res = {}
            for k in keys:
                cnt =  Counter([sent for fold in wrong_samples[k] for sent in fold])
                wrong = [re.sub(' ', '', k) for k,v in cnt.items() if v > 1]
                res[k] = wrong
            print('\n')
            print('over 2 times wrong samples:')
            print('\n')
            pprint(res)
            print("--- %s seconds ---" % (time.time() - start_time))
            if FOLDS_EPOCHS > 1:
                print('waiting 180 seconds...')
                time.sleep(180)
        return all_res
   

if __name__ == '__main__':
    # 读入数据
    df = pd.read_excel('./data/jd_comp_final_v5.xlsx')
    comp, non, hidden, not_hidden, equal, notequal, most_comp, hidden_index = read_x_data(df)

    # 一些参数（包含训练超参）
    DATASET = [not_hidden, non]
    SHOW_SAMPLES_CNT = 15
    MAX_WORDS = 100
    MODEL_NAME = 'HAN'
    CUT_MODE = 'simple'
    TEXT_FORMAT = 'text'
    ACTIVATION = 'selu' 
    BATCH_SIZE = 64
    N_EPOCHS = 6
    TEST_SIZE = 0.2
    NUM_LABELS = len(DATASET)
    EMBED_FILE = './data/embeddings/word_vectors_256d_20171228_5.txt'
    #EMBED_FILE = ''
    if EMBED_FILE:
        EMBED_TYPE = re.findall(r'(?<=/)\w+(?=_v)',EMBED_FILE)[0]
        EMBED_DIMS = int(re.findall(r'(?<=_)\d+(?=d)',EMBED_FILE)[0])
        PRE_TRAINED = True
    else:
        EMBED_TYPE = 'scratch'
        EMBED_DIMS = 256
        PRE_TRAINED = False
    N_FOLDS = 10
    FOLDS_EPOCHS = 1
    CV = False #是否进行交叉验证
    CHECK_HIDDEN = False #是否检查隐性比较句的错误情况
    ATTENTION_V = False #是否可视化attention权重
    SAVE_MODEL = False #是否保存模型结构
    PREDICT = False #是否预测新评论
    RAND = False
    
    

    # 整理数据格式
    x, y, s= get_x_y(DATASET, EMBED_TYPE)
    if PREDICT:
        df_predict = pd.read_excel('./data/jd_20w_v2.xlsx')
        predict_text = df_predict['segment'].tolist()
        total_text = x + predict_text
    total_text = x
    # reviews_length = np.array([len(sent.split()) for sent in x_copy])

    # 按空格切词，去除低频词
    tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower = True, split = " ")
    tokenizer.fit_on_texts(total_text)
    vocab = tokenizer.word_index
    vocab['UNK'] = 0
    word2idx = {v:k for k,v in vocab.items()}

    # 模型及权重保存路径
    DATE = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
    m_name = './model/' + MODEL_NAME + '_' + DATE + '.yaml'
    weights_name = './model/' + MODEL_NAME + '_weights_' + DATE + '.hdf5'

    # HAN模型需要的文本输入格式
    if MODEL_NAME in ['HAN','HMAN']:
        print('prepare inputs for HAN series model...')
        if EMBED_TYPE == 'word' or 'scratch':
            MAX_WORDS = 20
            MAX_SENTS = 5
        elif EMBED_TYPE == 'char':
            MAX_WORDS = 30
            MAX_SENTS = 6
            N_EPOCHS = 3
        N_LIMIT = MAX_WORDS * MAX_SENTS
        x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in x]
        if PREDICT:
            p_x = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in predict_text]
        TEXT_FORMAT = 'seq'
        new_name = MODEL_NAME + '_' + str(MAX_WORDS) + '_' + str(MAX_SENTS)
        m_name = './model/' + new_name + '_' + DATE + '.yaml'
        weights_name = './model/' + new_name + '_weights_' + DATE + '.hdf5'
    # 读入预训练的词向量矩阵
    if PRE_TRAINED and MODEL_NAME != 'one-hot':
        print('loading word embeddings...')
        embedding_matrix = load_embeddings(EMBED_FILE, vocab, EMBED_DIMS)
    else:
        embedding_matrix = None

    # 单次训练还是交叉验证训练
    print(EMBED_TYPE + ' model ' + MODEL_NAME + ' start training...')
    #time.sleep(3600)
    '''
    r_lst = [[1,1],[3,3],[5,5],[8,5],[10,5]]
    epoch_info = {}
    for i in range(5):
        tmp  = {}
        R = r_lst[i]
        print(R)
        result = train(CV, x, y, tokenizer, DATE)
        tmp['val_acc'] = result[4].history['val_acc']
        tmp['val_loss'] = result[4].history['val_loss']
        epoch_info[str(R[0])+'-'+str(R[1])] = tmp
    '''
    result = train(CV, x, y, tokenizer, DATE)
    show_text = read_line_data('./data/reviews_example.txt')
    xx = [split_sent(sent, MAX_WORDS, MAX_SENTS, CUT_MODE) for sent in show_text]
    xxx = np.array(get_sequences(tokenizer, xx, TEXT_FORMAT, MAX_WORDS))
    aa=visualize_attention(xxx,result[0],result[-1],result[-2], DATE, word2idx,0,False)
    if CHECK_HIDDEN:
        print(check_hidden(comp, result[3][1]))
        print(check_hidden(hidden, result[3][1]))
        print(check_hidden(equal, result[3][1]))
        print(check_hidden(notequal, result[3][1]))
    if not CV and PREDICT:
        print('predict labels for new reviews...')
        model = result[4]
        test = np.array(get_sequences(tokenizer, p_x, TEXT_FORMAT, MAX_WORDS))
        p_res = model.predict(test)
        df_predict['label_90'] = [1 if p[0] > 0.9 else 0 for p in p_res]
        df_predict[df_predict['label_90'] == 1]['cleaned_text'][0:10]
        df_predict.to_excel('./data/jd_16w_predict_res.xlsx', index = None)