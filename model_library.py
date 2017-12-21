import tensorflow as tf
from utils import split_sent
import numpy as np

from keras.layers import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers import MaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import LSTM, GRU, CuDNNGRU, Bidirectional
from keras.layers import TimeDistributed
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer

def embedding_layers(vocab_cnt, embed_size, max_len, embedding_matrix, pre_trained):
    if pre_trained:
        print('Using pretrained word embeddings...')
        return Embedding(vocab_cnt + 1, embed_size, input_length = max_len,
                         weights = [embedding_matrix], trainable = False)
    else:
        print('Using word embeddings from straching...')
        return Embedding(vocab_cnt + 1, embed_size, input_length = max_len)

def cnn_bn_net(conv_size, n_gram, padding_method, activation_func, last_layer):
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
        conv_net_1 = cnn_bn_net(conv_size[0], n_gram, padding_method,
                                activation_func, embed)
        conv_net_2 = cnn_bn_net(conv_size[1], n_gram, padding_method,
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
                  padding_way, drop_prob, fc_units, num_labels, activation_func,
                  classifier, loss_function, pre_trained, embedding_matrix):
    main_input = Input(shape=(max_len,), dtype='float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_len,
                             embedding_matrix, pre_trained)(main_input)
    cnn_list = []
    for ngram in ngram_filters:
        if len(ngram) == 1:
            conv = Convolution1D(conv_size[1], ngram[0],
                                 padding = padding_way)(embed)
            cnn_list.append(conv)
        else:
            conv1 = Convolution1D(conv_size[0], ngram[0],
                                  padding = padding_way)(embed)
            bn = BatchNormalization()(conv1)
            relu = Activation(activation_func)(bn)
            conv2 = Convolution1D(conv_size[1], ngram[1],
                                  padding = padding_way)(relu)
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


class AttLayer(Layer):
    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS, 1)
        # u: (MAX_TIMESTEPS, MAX_TIMESTEPS)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.dot(et, self.u))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0], input_shape[-1])

class Self_Att_Layer(Layer):
    def __init__(self, ws1, ws2, init = 'glorot_normal', **kwargs):
        self.kernel_initializer = initializers.get(init)
        self.weight_ws1 = ws1
        self.weight_ws2 = ws2
        super(Self_Att_Layer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.Ws1 = self.add_weight(shape = (input_shape[-1], self.weight_ws1),
                                 initializer = self.kernel_initializer,
                                 name = '{}_Ws1'.format(self.name))
        self.Ws2 = self.add_weight(shape = (self.weight_ws1, self.weight_ws2),
                                 initializer = self.kernel_initializer,
                                 name = '{}_Ws2'.format(self.name))
        super(Self_Att_Layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, input_shape, mask=None):
        H_reshaped = K.reshape(x, shape = [-1, input_shape[-1]])
        uit = K.tanh(K.dot(H_reshaped, self.Ws1))
        ait = K.reshape(K.dot(uit, self.Ws2),
                        shape = [input_shape[0], input_shape[1], -1])
        A = K.softmax(ait)
        A_T = K.permute_dimensions(A, (0,2,1))
        M_T = K.batch_dot(A_T, x)
        AA_T = K.batch_dot(A_T, A)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def SelfAtt(max_words, embed_size, vocab_cnt, gru_units,
            drop_prob, re_drop, num_labels, fc_units, classifier,
            loss_function, activation_func, pre_trained, embedding_matrix):
    sent_inputs = Input(shape=(max_words,), dtype = 'float64')
    sent_enc = Bidirectional(GRU(gru_units, dropout = drop_prob,
                                 recurrent_dropout = re_drop,
                                 return_sequences = True))(embed)
    sent_att = AttLayer()(sent_enc)
    fc = Dense(fc_units, activation = activation_func,
               kernel_initializer = 'he_normal')(sent_att)
    output = Dense(num_labels, activation = classifier)(fc)
    model = Model(inputs = sent_inputs, outputs = output)

    model.compile(loss = loss_function,
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    return model

def HAN(max_words, max_sents, embed_size, vocab_cnt, gru_units,
        drop_prob, re_drop, num_labels, fc_units, classifier,
        loss_function, activation_func, pre_trained, embedding_matrix):
    sent_inputs = Input(shape=(max_words,), dtype = 'float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_words,
                             embedding_matrix, pre_trained)(sent_inputs)
    sent_enc = Bidirectional(GRU(gru_units[0], dropout = drop_prob[0],
                                 recurrent_dropout = re_drop[0],
                                 return_sequences = True))(embed)
    sent_att = AttLayer(name = 'AttLayer')(sent_enc)
    sent_model = Model(sent_inputs, sent_att)

    doc_inputs = Input(shape = (max_sents, max_words), dtype = 'float64')
    doc_emb = TimeDistributed(sent_model)(doc_inputs)
    doc_enc = Bidirectional(GRU(gru_units[1], dropout = drop_prob[1],
                                recurrent_dropout = re_drop[1],
                                return_sequences = True))(doc_emb)
    doc_att = AttLayer(name = 'AttLayer')(doc_enc)

    fc1_drop = Dropout(drop_prob[1])(doc_att)
    #fc1_bn = BatchNormalization()(doc_att)
    fc1 = Dense(fc_units, activation = activation_func,
                kernel_initializer = 'he_normal')(fc1_drop)
    fc2_drop = Dropout(drop_prob[2])(fc1)
    #fc2_bn = BatchNormalization()(fc1)
    doc_pred = Dense(num_labels, activation = classifier)(fc2_drop)

    model = Model(inputs = doc_inputs, outputs = doc_pred)

    model.compile(loss = loss_function,
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    return sent_model, model

'''n-gram model'''
def generate_ngram():
    pass

def fasttext(max_words, embed_size, vocab_cnt, num_labels, classifier,
             loss_function, pre_trained, embedding_matrix):
    model = Sequential()
    model.add(embedding_layers(vocab_cnt, embed_size, max_words,
              embedding_matrix, pre_trained))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_labels, activation = classifier))

    model.compile(loss = loss_function,
              optimizer = 'adam',
              metrics = ['accuracy'])
    return model

'''attention visualization'''
def cal_att_weights(output, att_w):
    eij = np.tanh(np.dot(output[0], att_w[0]) + att_w[1])
    eij = eij.reshape((eij.shape[0], eij.shape[1]))
    ai = np.exp(np.dot(eij, att_w[2]))
    weights = ai / np.sum(ai)
    return weights

def get_attention(sent_model, doc_model, sequences, topN = 5):
    sent_before_att = K.function([sent_model.layers[0].input, K.learning_phase()], 
                              [sent_model.layers[2].output])
    doc_before_att = K.function([doc_model.layers[0].input, K.learning_phase()], 
                             [doc_model.layers[2].output])
    cnt_reviews = sequences.shape[0]
    
    # 找到重要的分句
    doc_att_w = doc_model.layers[3].get_weights()
    doc_sub_att = doc_before_att([sequences, 0])
    doc_att = cal_att_weights(doc_sub_att, doc_att_w)
    doc_sub_max = np.argmax(doc_att, axis = 1)
    key_sub_sents = np.array([sequences[k][v] for k,v in enumerate(doc_sub_max)])

    # 找到重要的词
    sent_att_w = sent_model.layers[3].get_weights()
    sent_sub_att = sent_before_att([key_sub_sents, 0])
    sent_att = cal_att_weights(sent_sub_att, sent_att_w)
    sent_sub_top_max = [sent_att[i].argsort()[-topN:] for i in range(cnt_reviews)]
    actual_word_idx = [key_sub_sents[i][sent_sub_top_max[i]] for i in range(cnt_reviews)]
    
    # 导出这个评论每个词的权重
    sent_all_att = []
    for i in range(cnt_reviews):
        sent_each_att = sent_before_att([sequences[i], 0])
        sent_each_att = cal_att_weights(sent_each_att, sent_att_w)
        sent_each_att = sent_each_att.ravel()
        sent_all_att.append(sent_each_att)
    sent_all_att = np.array(sent_all_att)
    
    return sent_all_att, sent_att, doc_att, actual_word_idx

def char_word_HAN(max_words, max_sents, embed_size, vocab_cnt, gru_units,
                  drop_prob, re_drop, num_labels, fc_units, classifier,
                  loss_function, activation_func, pre_trained, embedding_matrix):
    word_sent_inputs = Input(shape=(max_words[0],), dtype='float64')
    word_embed = embedding_layers(vocab_cnt[0], embed_size, max_words[0],
                                  embedding_matrix[0], pre_trained)(word_sent_inputs)
    word_sent_enc = Bidirectional(GRU(gru_units[0], dropout=drop_prob[0],
                                      recurrent_dropout=re_drop[0],
                                      return_sequences=True))(word_embed)
    word_sent_att = AttLayer(name='AttLayer')(word_sent_enc)
    word_sent_model = Model(word_sent_inputs, word_sent_att)

    word_doc_inputs = Input(shape=(max_sents[0], max_words[0]), 
                            dtype='float64',
                            name='word_inputs')
    word_doc_emb = TimeDistributed(word_sent_model)(word_doc_inputs)
    word_doc_enc = Bidirectional(GRU(gru_units[1], dropout=drop_prob[1],
                                     recurrent_dropout=re_drop[1],
                                     return_sequences=True))(word_doc_emb)
    word_doc_att = AttLayer(name='AttLayer_word')(word_doc_enc)

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
    char_sent_att = AttLayer(name='AttLayer')(char_sent_enc)
    char_sent_model = Model(char_sent_inputs, char_sent_att)

    char_doc_inputs = Input(shape=(max_sents[1], max_words[1]), 
                            dtype='float64',
                            name='char_inputs')
    char_doc_emb = TimeDistributed(char_sent_model)(char_doc_inputs)
    char_doc_enc = Bidirectional(GRU(gru_units[1], dropout=drop_prob[1],
                                recurrent_dropout=re_drop[1],
                                return_sequences=True))(char_doc_emb)
    char_doc_att = AttLayer(name='AttLayer_char')(char_doc_enc)

    char_fc1_drop = Dropout(drop_prob[1])(char_doc_att)
    char_fc1 = Dense(fc_units, activation=activation_func,
                     kernel_initializer='he_normal')(char_fc1_drop)
    char_fc2_drop = Dropout(drop_prob[2])(char_fc1)
    
    merge_info = concatenate([word_fc2_drop, char_fc2_drop], axis=1)
    output = Dense(num_labels, activation=classifier, name = 'out')(merge_info)

    model = Model(inputs=[word_doc_inputs, char_doc_inputs], outputs=output)

    model.compile(loss = loss_function,
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    return model

def one_hot_mdoel(x, y, classifier):
    pass




    

