'''some useful deep learning tools'''

import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras.activations import softmax

def np_softmax(array):
    return np.exp(array) / np.sum(np.exp(array))

def custom_loss(y_true, y_pred, loss_p, p_coef=0.004):
    return K.binary_crossentropy(y_pred, y_true) + p_coef * loss_p

# Attention Layers
# Include:
''' 
    1. Basic Attention from paper "Hierarchical Attention Networks for Document Classification(2016)"
    2. Self Attention from paper "A Structured Self-Attentive Sentence Embedding(2017)"
'''

# Basic Attention
class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Self Attention
class Self_Attention(Layer):
    def __init__(self, ws1, ws2, punish, init='glorot_normal', **kwargs):
        self.kernel_initializer = initializers.get(init)
        self.weight_ws1 = ws1
        self.weight_ws2 = ws2
        self.punish = punish
        super(Self_Attention, self).__init__(** kwargs)

    def build(self, input_shape):
        self.Ws1 = self.add_weight(shape=(input_shape[-1], self.weight_ws1),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name='{}_Ws1'.format(self.name))
        self.Ws2 = self.add_weight(shape=(self.weight_ws1, self.weight_ws2),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name='{}_Ws2'.format(self.name))
        self.batch_size = input_shape[0]
        super(Self_Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.tanh(K.dot(x, self.Ws1))
        ait = K.dot(uit, self.Ws2)
        ait = K.permute_dimensions(ait, (0, 2, 1))
        A = softmax(ait, axis=1)
        M = K.batch_dot(A, x)
        if self.punish:
            A_T = K.permute_dimensions(A, (0, 2, 1))
            tile_eye = K.tile(K.eye(self.weight_ws2), [self.batch_size, 1])
            tile_eye = K.reshape(
                tile_eye, shape=[-1, self.weight_ws2, self.weight_ws2])
            AA_T = K.batch_dot(A, A_T) - tile_eye
            P = K.l2_normalize(AA_T, axis=(1, 2))
            return M, P
        else:
            return M

    def compute_output_shape(self, input_shape):
        if self.punish:
            out1 = (input_shape[0], self.weight_ws2, input_shape[-1])
            out2 = (input_shape[0], self.weight_ws2, self.weight_ws2)
            return [out1, out2]
        else:
            return (input_shape[0], self.weight_ws2, input_shape[-1])

''' Attention Weights Calculation '''

# TODO: Complete attention weights calculation for model SelfAtt
def cal_att_weights(output, att_w, model_name):
    if model_name == 'HAN':
        eij = np.tanh(np.dot(output[0], att_w[0]) + att_w[1])
        eij = np.dot(eij, att_w[2])
        eij = eij.reshape((eij.shape[0], eij.shape[1]))
        ai = np.exp(eij)
        weights = ai / np.sum(ai)
        return weights
    elif model_name in ['Self_Att', 'MHAN']:
        uit = np.tanh(np.dot(output[0], att_w[0]))
        ait = np.dot(uit, att_w[1])
        ait = np.transpose(ait, axes=[0, 2, 1])
        ai = np.exp(ait)
        weights = np.array([ai[i] / np.sum(ai[i]) for i in range(ai.shape[0])])
        sum_weights = np.sum(weights, axis=-2)
        return sum_weights      
        
def get_attention(sent_model, doc_model, sequences, model_name, topN=5):
    sent_before_att = K.function([sent_model.layers[0].input, K.learning_phase()],
                                 [sent_model.layers[2].output])
    cnt_reviews = sequences.shape[0]

    # 导出这个句子每个词的权重
    sent_att_w = sent_model.layers[3].get_weights()
    sent_all_att = []
    for i in range(cnt_reviews):
        sent_each_att = sent_before_att([sequences[i], 0])
        sent_each_att = cal_att_weights(sent_each_att, sent_att_w, model_name)
        sent_each_att = sent_each_att.ravel()
        sent_all_att.append(sent_each_att)
    sent_all_att = np.array(sent_all_att)
    if model_name in ['HAN', 'MHAN']:
        doc_before_att = K.function([doc_model.layers[0].input, K.learning_phase()],
                                    [doc_model.layers[2].output])
        # 找到重要的分句
        doc_att_w = doc_model.layers[3].get_weights()
        doc_sub_att = doc_before_att([sequences, 0])
        doc_att = cal_att_weights(doc_sub_att, doc_att_w, model_name)
        return sent_all_att, doc_att
    elif model_name == 'Self_Att':
        return sent_all_att

    # 找到重要的词
    '''
    doc_sub_max = np.argmax(doc_att, axis = 1)
    key_sub_sents = np.array([sequences[k][v] for k,v in enumerate(doc_sub_max)])
    sent_sub_att = sent_before_att([key_sub_sents, 0])
    sent_att = cal_att_weights(sent_sub_att, sent_att_w, model_name)
    sent_sub_top_max = [sent_att[i].argsort()[-topN:] for i in range(cnt_reviews)]
    actual_word_idx = [key_sub_sents[i][sent_sub_top_max[i]] for i in range(cnt_reviews)] 
    '''
    
