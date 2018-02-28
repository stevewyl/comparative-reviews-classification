'''
text classification model zoo
'''
import tensorflow as tf
import numpy as np
from utils import split_sent
from tools import Attention, Self_Attention
from tools import get_attention

from keras.layers import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Flatten, TimeDistributed
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, Dropout
from keras.layers import LSTM, GRU, Bidirectional
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers

# 模型初始化
class Classification_Model(object):
    def __init__(self, config, embeddings, ntags):
        self.config = config
        self.embeddings = embeddings
        self.ntags = ntags
        self.model = None

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath=filepath)

    def __getattr__(self, name):
        return getattr(self.model, name)

# 嵌入层
def embedding_layers(config, embeddings=None):
    if embeddings is None:
        print('Using word embeddings from straching...')
        embed = Embedding(input_dim=config.vocab_cnt+1,
                          output_dim=config.embed_size)
    else:
        print('Using pretrained word embeddings...')
        embed = Embedding(input_dim=config.vocab_cnt+1,
                          output_dim=config.embed_size,
                          weights=[embeddings])
    return embed

class HAN(Classification_Model):
    def __init__(self, config, model_name='HAN', embeddings=None):
        self.model_name = model_name
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        doc_inputs = Input(shape=(config.max_sents, config.max_words), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码
        sent_enc = Bidirectional(GRU(config.rnn_units[0], dropout=config.drop_prob[0],
                                      recurrent_dropout=config.re_drop[0],
                                      return_sequences=True))(embed)
        sent_att = Attention(config.att_size[0], name='AttLayer')(sent_enc)
        self.sent_model = Model(sent_inputs, sent_att)
        # 段落编码
        doc_emb = TimeDistributed(self.sent_model)(doc_inputs)
        doc_enc = Bidirectional(GRU(config.rnn_units[1], dropout=config.drop_prob[1],
                                     recurrent_dropout=config.re_drop[1],
                                     return_sequences=True))(doc_emb)
        doc_att = Attention(config.att_size[1], name='AttLayer')(doc_enc)
        # FC
        fc1_drop = Dropout(config.drop_rate[1])(doc_att)
        fc1_bn = BatchNormalization()(fc1_drop)
        fc1 = Dense(config.fc_units, activation=config.activation_func,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.01))(fc1_bn)
        fc2_drop = Dropout(config.drop_prob[2])(fc1)
        # 输出
        doc_pred = Dense(config.ntags, activation=config.classifier)(fc2_drop)
        # 最终模型
        self.model = Model(inputs=doc_inputs, outputs=doc_pred)
        self.config = config

    # 获取注意力权重
    def get_attentions(self, sequences):
        return get_attention(self.sent_model, self.model, sequences, self.model_name)


class SelfAtt(Classification_Model):
    def __init__(self, config, model_name='self_att', embeddings=None):
        self.model_name = model_name
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码
        sent_enc = Bidirectional(GRU(config.rnn_units[0], dropout=config.drop_rate[0],
                                      recurrent_dropout=config.re_drop[0],
                                      return_sequences=True))(embed)
        sent_att = Self_Attention(350, config.r, punish=False, name='SelfAttLayer')(sent_enc)
        # FC
        flat = Flatten()(sent_att)
        fc1 = Dense(config.fc_units, activation=config.activation_func,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.01))(flat)
        # 输出
        output = Dense(config.ntag, activation=config.classifier)(fc1)
        # 最终模型
        self.model = Model(inputs=sent_inputs, outputs=output)
        self.config = config

    def get_attentions(self, sequences):
        return get_attention(self.model, None, sequences, self.model_name)

class MHAN(Classification_Model):
    def __init__(self, config, model_name='MHAN', embeddings=None):
        self.model_name = model_name
        # 定义模型输入
        


def cnn_bn_block(conv_size, n_gram, padding_method, activation_func, last_layer):
    conv = Convolution1D(conv_size, n_gram, padding = padding_method)(last_layer)
    bn = BatchNormalization()(conv)
    relu = Activation(activation_func)(bn)
    return relu

def TextCNNBN(max_len, embed_size, vocab_cnt, ngram_filters, conv_size, pool_size,
              padding_method, fc_units, num_labels, activation_func, classifier,
              loss_function, pre_trained, embedding_matrix):
    main_input = Input(shape=(max_len,), dtype='float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_len,
                             embedding_matrix, pre_trained)(main_input)
    cnn_list = []
    for n_gram in ngram_filters:
        conv_net_1 = cnn_bn_block(conv_size[0], n_gram, padding_method,
                                activation_func, embed)
        conv_net_2 = cnn_bn_block(conv_size[1], n_gram, padding_method,
                                activation_func, conv_net_1)
        max_pool = MaxPool1D(pool_size)(conv_net_2)
        cnn_list.append(max_pool)
    cnn = concatenate(cnn_list, axis=-1)
    flat = Flatten()(cnn)
    fc_1 = Dense(fc_units, kernel_initializer = 'he_normal')(flat)
    bn = BatchNormalization()(fc_1)
    relu = Activation(activation_func)(bn)
    main_output = Dense(num_labels, activation = classifier)(relu)
    model = Model(inputs = main_input, outputs = main_output)
    model.compile(loss = loss_function,
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    return model


def TextInception(max_len, embed_size, vocab_cnt, ngram_filters, conv_size,
                  padding_method, drop_prob, fc_units, num_labels, activation_func,
                  classifier, loss_function, pre_trained, embedding_matrix):
    main_input = Input(shape=(max_len,), dtype='float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_len,
                             embedding_matrix, pre_trained)(main_input)
    cnn_list = []
    for ngram in ngram_filters:
        if len(ngram) == 1:
            conv = Convolution1D(conv_size[1], ngram[0],
                                 padding=padding_method)(embed)
            cnn_list.append(conv)
        else:
            conv1 = Convolution1D(conv_size[0], ngram[0],
                                  padding = padding_method)(embed)
            bn = BatchNormalization()(conv1)
            relu = Activation(activation_func)(bn)
            conv2 = Convolution1D(conv_size[1], ngram[1],
                                  padding=padding_method)(relu)
            cnn_list.append(conv2)

    inception = concatenate(cnn_list, axis=-1)

    flat = Flatten()(inception)
    fc = Dense(fc_units, kernel_initializer = 'he_normal')(flat)
    drop = Dropout(drop_prob)(fc)
    bn = BatchNormalization()(drop)
    relu = Activation(activation_func)(bn)
    main_output = Dense(num_labels, activation = classifier)(relu)
    model = Model(inputs = main_input, outputs = main_output)
    model.compile(loss = loss_function,
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    return model

# 有bug
def convRNN(max_len, embed_size, vocab_cnt, hidden_units, drop_rate,
            conv_size, filter_size, padding_method, pool_size, fc_units,
            activation_fun, num_labels, classifier, loss_function,
            pre_trained, embedding_matrix):
    main_input = Input(shape=(max_len,), dtype='float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_len,
                             embedding_matrix, pre_trained)(main_input)
    forward = GRU(hidden_units, return_sequences = True,
                  recurrent_dropout = drop_rate)(embed)
    backward = GRU(hidden_units, return_sequences = True, go_backwards = True,
                   recurrent_dropout = drop_rate)(embed)
    _, last_state_for = Lambda(lambda x: tf.split(x, [max_len-1, 1], 1))(forward)
    _, last_state_back = Lambda(lambda x: tf.split(x, [max_len-1, 1], 1))(backward)
    bi_rnn_output = concatenate([forward, backward], axis = -1)
    cnn_1 = Convolution1D(conv_size[0], filter_size,
                          padding = padding_method)(bi_rnn_output)
    bn = BatchNormalization()(cnn_1)
    cnn_2 = Convolution1D(conv_size[1], filter_size, padding = padding_method)(bn)
    max_pool = MaxPool1D(pool_size = pool_size)(cnn_2)
    final = concatenate([last_state_for, max_pool, last_state_back], axis = 1)
    flat = Flatten()(final)
    relu = Dense(fc_units, activation = activation_fun,
                 kernel_initializer = 'he_normal')(flat)
    main_output = Dense(num_labels, activation = classifier)(relu)
    model = Model(inputs = main_input, outputs = main_output)
    model.compile(loss = loss_function,
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    return model

def HMAN(r, max_words, max_sents, embed_size, vocab_cnt, gru_units,
         drop_prob, re_drop, num_labels, fc_units, classifier,
         loss_function, activation_func, pre_trained, embedding_matrix):
    sent_inputs = Input(shape=(max_words,), dtype = 'float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_words,
                             embedding_matrix, pre_trained)(sent_inputs)
    sent_enc = Bidirectional(GRU(gru_units[0], dropout = drop_prob[0],
                                 recurrent_dropout = re_drop[0],
                                 return_sequences = True))(embed)
    sent_att = Self_Attention(150, r[0], False)(sent_enc)
    sent_flat = Flatten()(sent_att)
    sent_model = Model(sent_inputs, sent_flat)
    
    doc_inputs = Input(shape = (max_sents, max_words), dtype = 'float64')
    doc_emb = TimeDistributed(sent_model)(doc_inputs)
    doc_enc = Bidirectional(LSTM(gru_units[1], dropout = drop_prob[1],
                            recurrent_dropout = re_drop[1],
                            return_sequences = True))(doc_emb)
    doc_att = Self_Attention(50, r[1], False)(doc_enc)
    doc_flat = Flatten()(doc_att)
    fc = Dense(fc_units, activation = activation_func,
               kernel_initializer = 'he_normal',
               kernel_regularizer=regularizers.l2(0.01))(doc_flat)
    output = Dense(num_labels, activation = classifier)(fc)
    model = Model(inputs = doc_inputs, outputs = output)
    
    opt = optimizers.Nadam(clipnorm=1.)
    model.compile(loss = loss_function,
                  optimizer = opt,
                  metrics = ['accuracy'])
    return sent_model, model




def char_word_HAN(max_words, max_sents, embed_size, vocab_cnt, gru_units,
                  drop_prob, att_size, re_drop, num_labels, fc_units, classifier,
                  loss_function, activation_func, pre_trained, embedding_matrix):
    word_sent_inputs = Input(shape=(max_words[0],), dtype='float64')
    word_embed = embedding_layers(vocab_cnt[0], embed_size, max_words[0],
                                  embedding_matrix[0], pre_trained)(word_sent_inputs)
    word_sent_enc = Bidirectional(GRU(gru_units[0], dropout=drop_prob[0],
                                      recurrent_dropout=re_drop[0],
                                      return_sequences=True))(word_embed)
    word_sent_att = Attention(att_size[0], name='AttLayer')(word_sent_enc)
    word_sent_model = Model(word_sent_inputs, word_sent_att)

    word_doc_inputs = Input(shape=(max_sents[0], max_words[0]), 
                            dtype='float64',
                            name='word_inputs')
    word_doc_emb = TimeDistributed(word_sent_model)(word_doc_inputs)
    word_doc_enc = Bidirectional(GRU(gru_units[1], dropout=drop_prob[1],
                                     recurrent_dropout=re_drop[1],
                                     return_sequences=True))(word_doc_emb)
    word_doc_att = Attention(att_size[1], name='AttLayer_word')(word_doc_enc)

    word_fc1_drop = Dropout(drop_prob[1])(word_doc_att)
    word_fc1 = Dense(fc_units, activation=activation_func,
                     kernel_initializer='he_normal')(word_fc1_drop)
    word_fc2_drop = Dropout(drop_prob[2])(word_fc1)

    char_sent_inputs = Input(shape=(max_words[1],), dtype='float64')
    char_embed = embedding_layers(vocab_cnt[1], embed_size, max_words[1],
                                  embedding_matrix[1], pre_trained)(char_sent_inputs)
    char_sent_enc = Bidirectional(GRU(gru_units[0], dropout=drop_prob[0],
                                      recurrent_dropout=re_drop[0],
                                      return_sequences=True))(char_embed)
    char_sent_att = Attention(att_size[2], name='AttLayer')(char_sent_enc)
    char_sent_model = Model(char_sent_inputs, char_sent_att)

    char_doc_inputs = Input(shape=(max_sents[1], max_words[1]), 
                            dtype='float64',
                            name='char_inputs')
    char_doc_emb = TimeDistributed(char_sent_model)(char_doc_inputs)
    char_doc_enc = Bidirectional(GRU(gru_units[1], dropout=drop_prob[1],
                                recurrent_dropout=re_drop[1],
                                return_sequences=True))(char_doc_emb)
    char_doc_att = Attention(att_size[3], name='AttLayer_char')(char_doc_enc)

    char_fc1_drop = Dropout(drop_prob[1])(char_doc_att)
    char_fc1 = Dense(fc_units, activation=activation_func,
                     kernel_initializer='he_normal')(char_fc1_drop)
    char_fc2_drop = Dropout(drop_prob[2])(char_fc1)
    
    merge_info = concatenate([word_fc2_drop, char_fc2_drop], axis=1)
    output = Dense(num_labels, activation=classifier, name = 'out')(merge_info)

    model = Model(inputs=[word_doc_inputs, char_doc_inputs], outputs=output)
    
    nadam = optimizers.Nadam(clipnorm=1.)
    model.compile(loss = loss_function,
                  optimizer =  nadam,
                  metrics = ['accuracy'])
    return model

def sent_word_HAN(add_shape, max_words, max_sents, embed_size, vocab_cnt, gru_units,
                  drop_prob, att_size, re_drop, num_labels, fc_units, classifier,
                  loss_function, activation_func, pre_trained, embedding_matrix):
    sent_inputs = Input(shape=(max_words,), dtype = 'float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_words,
                             embedding_matrix, pre_trained)(sent_inputs)
    sent_enc = Bidirectional(GRU(gru_units[0], dropout = drop_prob[0],
                                 recurrent_dropout = re_drop[0],
                                 return_sequences = True))(embed)
    sent_att = Attention(att_size[0], name='AttLayer')(sent_enc)
    sent_model = Model(sent_inputs, sent_att)
    
    doc_inputs = Input(shape = (max_sents, max_words), dtype = 'float64', name = 'text_inputs')
    addtional_inputs = Input(shape = (add_shape,), name = 'addtional_inputs')
    doc_emb = TimeDistributed(sent_model)(doc_inputs)
    doc_enc = Bidirectional(GRU(gru_units[1], dropout = drop_prob[1],
                                recurrent_dropout = re_drop[1],
                                return_sequences = True))(doc_emb)
    
    doc_att = Attention(att_size[1], name='AttLayer')(doc_enc)
    sent_output = Dense(num_labels, activation = classifier, name = 'att_output')(doc_att)
    new_tensor = concatenate([doc_att, addtional_inputs])
    fc1_drop = Dropout(drop_prob[1])(new_tensor)
    #fc1_bn = BatchNormalization()(doc_att)
    fc1 = Dense(fc_units, activation = activation_func,
                kernel_initializer = 'he_normal',
                kernel_regularizer=regularizers.l2(0.01))(fc1_drop)
    fc2_drop = Dropout(drop_prob[2])(fc1)
    #fc2_bn = BatchNormalization()(fc1)
    doc_pred = Dense(num_labels, activation = classifier, name = 'model_output')(fc2_drop)

    model = Model(inputs = [doc_inputs, addtional_inputs], outputs = [doc_pred, sent_output])
    opt = optimizers.Adam(clipnorm=1.)
    model.compile(loss = {'model_output':loss_function, 'att_output':loss_function},
                  optimizer = opt,
                  loss_weights = [1., 0.2],
                  metrics = ['accuracy'])
    return sent_model, model


def one_hot_mdoel(vocab_cnt, drop_rate, fc_units, num_labels, activation_func,
                  classifier, loss_func):
    model = Sequential()
    model.add(Dense(fc_units, input_shape=(vocab_cnt,), activation=activation_func,
                    kernel_initializer = 'he_normal',
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(drop_rate))
    model.add(Dense(num_labels,activation=classifier))
    opt = optimizers.Adam(clipnorm=1.)
    model.compile(loss=loss_func,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

'''n-gram model'''
def generate_ngram():
    pass

def fasttext(vocab_cnt, max_words, embed_size, embedding_matrix, 
             pre_trained, num_labels, classifier, loss_func):
    model = Sequential()
    model.add(embedding_layers(vocab_cnt, embed_size, max_words,
                             embedding_matrix, pre_trained))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_labels, activation=classifier))
    
    opt = optimizers.Adam(clipnorm=1.)
    model.compile(loss=loss_func,
                  optimizer= opt ,
                  metrics=['accuracy'])
    return model

def bi_rnn(vocab_cnt, embed_size, max_words, embedding_matrix, pre_trained, 
           gru_units, drop_rate, num_labels, classifier, loss_func):
    model = Sequential()
    model.add(embedding_layers(vocab_cnt, embed_size, max_words,
                               embedding_matrix, pre_trained))
    model.add(GRU(gru_units, dropout=drop_rate[0], 
                  recurrent_dropout=drop_rate[1], return_sequences=True))
    model.add(GRU(gru_units, dropout=drop_rate[0], recurrent_dropout=drop_rate[1]))
    model.add(Dense(num_labels, activation=classifier))
    opt = optimizers.Adam(clipnorm=1.)
    model.compile(loss=loss_func,
                  optimizer= opt ,
                  metrics=['accuracy'])
    return model

def TextCNN(vocab_cnt, embed_size, max_words, embedding_matrix, pre_trained,
            cnn_size, filter_size, stride, pool_size, drop_rate, num_labels,
            classifier, loss_func, fc_units):
    main_input = Input(shape=(max_words,), dtype='float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_words,
                                embedding_matrix, pre_trained)(main_input)
    cnn1 = Convolution1D(cnn_size, filter_size[0], padding='same', 
                         strides = stride, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=pool_size[0])(cnn1)
    cnn2 = Convolution1D(cnn_size, filter_size[1], padding='same', 
                         strides = stride, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=pool_size[1])(cnn2)
    cnn3 = Convolution1D(cnn_size, filter_size[2], padding='same', 
                         strides = stride, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=pool_size[2])(cnn3)
    cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(drop_rate)(flat)
    fc = Dense(fc_units, activation='relu', kernel_initializer = 'he_normal',
               kernel_regularizer=regularizers.l2(0.01))(drop)

    main_output = Dense(num_labels, activation=classifier)(fc)
    model = Model(inputs = main_input, outputs = main_output)
    opt = optimizers.Adam(clipnorm=1.)
    model.compile(loss=loss_func,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model





    


    

