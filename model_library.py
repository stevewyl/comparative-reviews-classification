'''
text classification model zoo
bug: graphviz has not been installed, remove model visualization fuction temporarily
'''
import tensorflow as tf
import numpy as np
from layers import Attention, Self_Attention
from layers import get_attention

from keras.layers import concatenate
from keras.models import Sequential, Model, model_from_yaml
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Flatten, TimeDistributed
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, Dropout
from keras.layers import LSTM, GRU, Bidirectional
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers
#from keras.utils import plot_model


# 模型初始化
class Classification_Model(object):
    def __init__(self, config, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.model_name = config.model_name
        self.model = None

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save_model(self, filepath):
        yaml_string  = self.model.to_yaml()
        with open(filepath, 'w') as f:
            f.write(yaml_string)

    def load_model(self, filepath):
        return model_from_yaml(filepath)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath=filepath)
    '''
    def plot(self, filepath):
        plot_model(self.model, to_file = filepath)
    '''
    
# 嵌入层
def embedding_layers(config, embeddings=None):
    if embeddings is None:
        print('Using word embeddings from straching...')
        embed = Embedding(input_dim=config.vocab_cnt,
                          output_dim=config.embed_size)
    else:
        print('Using pretrained word embeddings...')
        embed = Embedding(input_dim=config.vocab_cnt,
                          output_dim=config.embed_size,
                          weights=[embeddings])
    return embed

# 单注意力层次网络
class HAN(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        doc_inputs = Input(shape=(config.max_sents, config.max_words), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码
        sent_enc = Bidirectional(GRU(config.rnn_units[0], dropout=config.drop_rate[0],
                                     recurrent_dropout=config.re_drop[0],
                                     return_sequences=True))(embed)
        sent_att = Attention(config.att_size[0], name='AttLayer')(sent_enc)
        self.sent_model = Model(sent_inputs, sent_att)
        # 段落编码
        doc_emb = TimeDistributed(self.sent_model)(doc_inputs)
        doc_enc = Bidirectional(GRU(config.rnn_units[1], dropout=config.drop_rate[1],
                                    recurrent_dropout=config.re_drop[1],
                                    return_sequences=True))(doc_emb)
        doc_att = Attention(config.att_size[1], name='AttLayer')(doc_enc)
        # FC
        fc1_drop = Dropout(config.drop_rate[1])(doc_att)
        fc1_bn = BatchNormalization()(fc1_drop)
        fc1 = Dense(config.fc_units[0], activation=config.activation_func,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.01))(fc1_bn)
        fc2_drop = Dropout(config.drop_rate[1])(fc1)
        # 输出
        doc_pred = Dense(config.ntags, activation=config.classifier)(fc2_drop)
        # 最终模型
        self.model = Model(inputs=doc_inputs, outputs=doc_pred)
        self.config = config

    # 获取注意力权重
    def get_attentions(self, sequences):
        return get_attention(self.sent_model, self.model, sequences, self.model_name)

# 多注意力网络
class SelfAtt(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码
        sent_enc = Bidirectional(GRU(config.rnn_units[0], dropout=config.drop_rate[0],
                                      recurrent_dropout=config.re_drop[0],
                                      return_sequences=True))(embed)
        sent_att = Self_Attention(config.ws1, config.r, punish=False, name='SelfAttLayer')(sent_enc)
        # FC
        flat = Flatten()(sent_att)
        fc = Dense(config.fc_units[0], activation=config.activation_func,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.01))(flat)
        # 输出
        output = Dense(config.ntags, activation=config.classifier)(fc)
        # 最终模型
        self.model = Model(inputs=sent_inputs, outputs=output)
        self.config = config

    def get_attentions(self, sequences):
        return get_attention(self.model, None, sequences, self.model_name)

# 多注意力层次网络
class MHAN(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        doc_inputs = Input(shape=(config.max_sents, config.max_words), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码
        sent_enc = Bidirectional(GRU(config.rnn_units[0], dropout=config.drop_rate[0],
                                      recurrent_dropout=config.re_drop[0],
                                      return_sequences=True))(embed)
        sent_att = Self_Attention(config.ws1[0], config.r[0], False, name='SelfAttLayer')(sent_enc)
        sent_flat = Flatten()(sent_att)
        self.sent_model = Model(sent_inputs, sent_flat)
        # 段落编码
        doc_emb = TimeDistributed(self.sent_model)(doc_inputs)
        doc_enc = Bidirectional(GRU(config.rnn_units[1], dropout=config.drop_rate[1],
                                    recurrent_dropout=config.re_drop[1],
                                    return_sequences=True))(doc_emb)
        doc_att = Self_Attention(config.ws1[1], config.r[1], False, name='SelfAttLayer')(doc_enc)
        # FC
        doc_flat = Flatten()(doc_att)
        fc = Dense(config.fc_units[0], activation=config.activation_func,
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01))(doc_flat)
        # 输出
        output = Dense(config.ntags, activation=config.classifier)(fc)
        # 最终模型
        self.model = Model(inputs=doc_inputs, outputs=output)
        self.config = config
    
    def get_attentions(self, sequences):
        return get_attention(self.sent_model, self.model, sequences, self.model_name)


# 双向RNN模型
class Bi_RNN(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)   
        # 句子编码
        sent_enc = Bidirectional(GRU(config.rnn_units[0], 
                                     dropout=config.drop_rate[0],
                                     recurrent_dropout=config.re_drop[0],
                                     return_sequences=True))(embed)
        # 输出
        output = Dense(config.ntags, activation=config.classifier)(sent_enc)
        # 最终模型
        self.model = Model(inputs=sent_inputs, outputs=output)
        self.config = config

# 多通道CNN模型
class TextCNN(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码
        cnn_combine = []
        for f,p in zip(config.filter_size, config.pool_size):
            cnn = Convolution1D(config.conv_size, f, padding='same')(embed)
            pool = MaxPool1D(pool_size=p)(cnn)
            cnn_combine.append(pool)
        cnn_all = concatenate(cnn_combine, axis=-1)
        # FC
        flat = Flatten()(cnn_all)
        drop = Dropout(config.drop_rate[0])(flat)
        fc = Dense(config.fc_units[0], activation=config.activation_func, 
                   kernel_initializer = 'he_normal',
                   kernel_regularizer=regularizers.l2(0.01))(drop)
        # 输出
        output = Dense(config.ntags, activation=config.classifier)(fc)
        # 最终模型
        self.model = Model(inputs=sent_inputs, outputs=output)
        self.config = config

class TextCNNBN(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码
        cnn_combine = []
        for f,p in zip(config.filter_size, config.pool_size):
            conv_net_1 = Convolution1D(config.conv_size[0], f, padding='same')(embed)
            bn_1 = BatchNormalization()(conv_net_1)
            relu_1 = Activation(config.activation_func)(bn_1)
            conv_net_2 = Convolution1D(config.conv_size[1], f, padding='same')(relu_1)
            bn_2 = BatchNormalization()(conv_net_2)
            relu_2 = Activation(config.activation_func)(bn_2)
            pool = MaxPool1D(pool_size=p)(relu_2)
            cnn_combine.append(pool)
        cnn_all = concatenate(cnn_combine, axis=-1)
        # FC
        flat = Flatten()(cnn_all)
        fc = Dense(config.fc_units[0], kernel_initializer='he_normal')(flat)
        bn = BatchNormalization()(fc)
        relu = Activation(config.activation_func)(bn)
        # 输出
        output = Dense(config.ntags, activation=config.classifier)(relu)
        # 最终模型
        self.model = Model(inputs=sent_inputs, outputs=output)
        self.config = config


class TextInception(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码
        cnn_combine = []
        for f in config.filter_size:
            if len(f) == 1:
                conv = Convolution1D(config.conv_size[1], f[0],
                                     padding='same')(embed)
                cnn_combine.append(conv)
            else:
                conv_net_1 = Convolution1D(config.conv_size[0], f[0],
                                           padding='same')(embed)
                bn = BatchNormalization()(conv_net_1)
                relu = Activation(config.activation_func)(bn)
                conv_net_2 = Convolution1D(config.conv_size[1], f[1], 
                                           padding='same')(relu)
                cnn_combine.append(conv_net_2)
        inception = concatenate(cnn_combine, axis=-1)
        # FC
        flat = Flatten()(inception)
        bn = BatchNormalization()(flat)
        relu = Activation(config.activation_func)(bn)
        # 输出
        output = Dense(config.ntags, activation=config.classifier)(relu)
        # 最终模型
        self.model = Model(inputs=sent_inputs, outputs=output)
        self.config = config

class convRNN(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码（RNN)
        forward = GRU(config.rnn_units[0], return_sequences=True,
                      recurrent_dropout=config.drop_rate[0])(embed)
        backward = GRU(config.rnn_units[0], return_sequences=True, go_backwards=True, 
                       recurrent_dropout=config.drop_rate[0])(embed)
        _, last_state_for = Lambda(lambda x: tf.split(x, [config.max_words-1, 1], 1))(forward)
        _, last_state_back = Lambda(lambda x: tf.split(x, [config.max_words-1, 1], 1))(backward)
        bi_rnn_output = concatenate([forward, backward], axis = -1)
        # 句子编码（CNN)
        conv_net_1 = Convolution1D(config.conv_size[0], config.filter_size, 
                                   padding='same')(bi_rnn_output)
        bn = BatchNormalization()(conv_net_1)
        conv_net_2 = Convolution1D(config.conv_size[1], config.filter_size, 
                                   padding='same')(bn)
        pool = MaxPool1D(config.pool_size[0])(conv_net_2)
        final = concatenate([last_state_for, pool, last_state_back], axis = 1)
        # FC
        flat = Flatten()(final)
        fc = Dense(config.fc_units[0], activation=config.activation_func)(flat)
        # 输出
        output = Dense(config.ntags, activation=config.classifier)(fc)   
        # 最终模型
        self.model = Model(inputs=sent_inputs, outputs=output)
        self.config = config

# fasttext
# from paper "Bag of Tricks for Efficient Text Classification(2016)"
class fasttext(Classification_Model):
    def __init__(self, config, embeddings=None):
        # 定义模型输入
        sent_inputs = Input(shape=(config.max_words,), dtype='float64')
        # 嵌入层
        embed = embedding_layers(config, embeddings)(sent_inputs)
        # 句子编码        
        pool = GlobalAveragePooling1D()(embed)
        # 输出
        output = Dense(config.ntags, activation=config.classifier)(pool)
        # 最终模型
        self.model = Model(inputs=sent_inputs, outputs=output)
        self.config = config


'''
Test model
'''
def char_word_HAN(max_words, max_sents, embed_size, vocab_cnt, gru_units,
                  drop_rate, att_size, re_drop, num_labels, fc_units, classifier,
                  loss_function, activation_func, pre_trained, embedding_matrix):
    word_sent_inputs = Input(shape=(max_words[0],), dtype='float64')
    word_embed = embedding_layers(vocab_cnt[0], embed_size, max_words[0],
                                  embedding_matrix[0], pre_trained)(word_sent_inputs)
    word_sent_enc = Bidirectional(GRU(gru_units[0], dropout=drop_rate[0],
                                      recurrent_dropout=re_drop[0],
                                      return_sequences=True))(word_embed)
    word_sent_att = Attention(att_size[0], name='AttLayer')(word_sent_enc)
    word_sent_model = Model(word_sent_inputs, word_sent_att)

    word_doc_inputs = Input(shape=(max_sents[0], max_words[0]), 
                            dtype='float64',
                            name='word_inputs')
    word_doc_emb = TimeDistributed(word_sent_model)(word_doc_inputs)
    word_doc_enc = Bidirectional(GRU(gru_units[1], dropout=drop_rate[1],
                                     recurrent_dropout=re_drop[1],
                                     return_sequences=True))(word_doc_emb)
    word_doc_att = Attention(att_size[1], name='AttLayer_word')(word_doc_enc)

    word_fc1_drop = Dropout(drop_rate[1])(word_doc_att)
    word_fc1 = Dense(fc_units, activation=activation_func,
                     kernel_initializer='he_normal')(word_fc1_drop)
    word_fc2_drop = Dropout(drop_rate[2])(word_fc1)

    char_sent_inputs = Input(shape=(max_words[1],), dtype='float64')
    char_embed = embedding_layers(vocab_cnt[1], embed_size, max_words[1],
                                  embedding_matrix[1], pre_trained)(char_sent_inputs)
    char_sent_enc = Bidirectional(GRU(gru_units[0], dropout=drop_rate[0],
                                      recurrent_dropout=re_drop[0],
                                      return_sequences=True))(char_embed)
    char_sent_att = Attention(att_size[2], name='AttLayer')(char_sent_enc)
    char_sent_model = Model(char_sent_inputs, char_sent_att)

    char_doc_inputs = Input(shape=(max_sents[1], max_words[1]), 
                            dtype='float64',
                            name='char_inputs')
    char_doc_emb = TimeDistributed(char_sent_model)(char_doc_inputs)
    char_doc_enc = Bidirectional(GRU(gru_units[1], dropout=drop_rate[1],
                                recurrent_dropout=re_drop[1],
                                return_sequences=True))(char_doc_emb)
    char_doc_att = Attention(att_size[3], name='AttLayer_char')(char_doc_enc)

    char_fc1_drop = Dropout(drop_rate[1])(char_doc_att)
    char_fc1 = Dense(fc_units, activation=activation_func,
                     kernel_initializer='he_normal')(char_fc1_drop)
    char_fc2_drop = Dropout(drop_rate[2])(char_fc1)
    
    merge_info = concatenate([word_fc2_drop, char_fc2_drop], axis=1)
    output = Dense(num_labels, activation=classifier, name = 'out')(merge_info)

    model = Model(inputs=[word_doc_inputs, char_doc_inputs], outputs=output)
    
    nadam = optimizers.Nadam(clipnorm=1.)
    model.compile(loss = loss_function,
                  optimizer =  nadam,
                  metrics = ['accuracy'])
    return model

def sent_word_HAN(add_shape, max_words, max_sents, embed_size, vocab_cnt, gru_units,
                  drop_rate, att_size, re_drop, num_labels, fc_units, classifier,
                  loss_function, activation_func, pre_trained, embedding_matrix):
    sent_inputs = Input(shape=(max_words,), dtype = 'float64')
    embed = embedding_layers(vocab_cnt, embed_size, max_words,
                             embedding_matrix, pre_trained)(sent_inputs)
    sent_enc = Bidirectional(GRU(gru_units[0], dropout = drop_rate[0],
                                 recurrent_dropout = re_drop[0],
                                 return_sequences = True))(embed)
    sent_att = Attention(att_size[0], name='AttLayer')(sent_enc)
    sent_model = Model(sent_inputs, sent_att)
    
    doc_inputs = Input(shape = (max_sents, max_words), dtype = 'float64', name = 'text_inputs')
    addtional_inputs = Input(shape = (add_shape,), name = 'addtional_inputs')
    doc_emb = TimeDistributed(sent_model)(doc_inputs)
    doc_enc = Bidirectional(GRU(gru_units[1], dropout = drop_rate[1],
                                recurrent_dropout = re_drop[1],
                                return_sequences = True))(doc_emb)
    
    doc_att = Attention(att_size[1], name='AttLayer')(doc_enc)
    sent_output = Dense(num_labels, activation = classifier, name = 'att_output')(doc_att)
    new_tensor = concatenate([doc_att, addtional_inputs])
    fc1_drop = Dropout(drop_rate[1])(new_tensor)
    #fc1_bn = BatchNormalization()(doc_att)
    fc1 = Dense(fc_units, activation = activation_func,
                kernel_initializer = 'he_normal',
                kernel_regularizer=regularizers.l2(0.01))(fc1_drop)
    fc2_drop = Dropout(drop_rate[2])(fc1)
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







    


    

