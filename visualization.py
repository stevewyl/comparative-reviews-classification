import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import math
import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.metrics import roc_curve, auc

# 简单绘制准确率和loss曲线
def plot_loss_accuray(history):
    plt.subplot(211)
    plt.title("Training and validation accuracy")
    plt.plot(history.history["acc"], color = "g", label = "Train")
    plt.plot(history.history["val_acc"], color = "b", label = "Test")
    plt.legend(loc = "best")
    
    plt.subplot(212)
    plt.title("Training and validation loss")
    plt.plot(history.history["loss"], color = "g", label = "Train")
    plt.plot(history.history["val_loss"], color = "b", label = "Test")
    plt.legend(loc = "best")
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_score, n_classes, index):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[index], tpr[index], color = 'darkorange',
             lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc[index])
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc = "lower right")
    plt.show()

def annotated_heatmap(values, text):
    fig = ff.create_annotated_heatmap(values, 
                                      annotation_text=text, 
                                      colorscale='Hot', 
                                      hoverinfo='text')
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 8
    plot(fig, image_width = 1500, image_height = 300)

def customed_heatmap(all_att, text_sent, n_limit, date, label, text_type):
    if text_type == 'sent':
        z, symbol = [], []
        for i in range(len(text_sent)):
            idx_text = {k:word for k,word in enumerate(text_sent[i]) if word != 'UNK'}
            idx = list(idx_text.keys())
            text = list(idx_text.values())
            value = list(all_att[i][idx])
            num_words = len(text)
            total_words = math.ceil(num_words/float(n_limit))*n_limit
            text = text + [' '] * (total_words-num_words)
            value = value + [0] * (total_words-num_words)
            z.append(value)
            symbol.append(text)
        hover = symbol
        colorscale = [[0.0, '#FFFFFF'],[.5, '#00BFFF'], 
                      [.75, '#00008B'],[1.0, '#191970']]
        pt = ff.create_annotated_heatmap(z, annotation_text=symbol, text=hover,
                                         colorscale=colorscale, 
                                         font_colors=['black'], 
                                         hoverinfo='text')
        for i in range(len(pt.layout.annotations)):
            pt.layout.annotations[i].font.size = 10
        if label == 0:
            file_name = './result/sent_attention_visualization_' + date + '.html'
            plot(pt, filename = file_name, image_width = 1200, image_height = 200)
        else:
            plot(pt, image_width = 1200, image_height = 200)
    elif text_type == 'doc':
        
        colorscale = [[0.0, '#FFFFFF'],[.25, '#EEB4B4'], 
                      [.5, '#EE6A50'],[1.0, '#EE0000']]
        x_label = ['DOC_' + str(i+1) for i in range(all_att.shape[1])]
        y_label = ['Sub_sent_' + str(i+1) for i in range(all_att.shape[0])][::-1]
        data = [
            go.Heatmap(
                z=all_att,
                x=x_label,
                y=y_label,
                colorscale=colorscale,
            )
        ]
        fig = go.Figure(data=data)
        if label == 0:
            file_name = './result/doc_attention_visualization_' + date + '.html'
            plot(fig, filename=file_name)
        else:
            plot(fig)

# 可视化attention权重
def visualize_attention(text, att, date, word2idx, model_name, n_limit, label):
    normalizer_sent = Normalizer()
    cnt = text.shape[0]

    if model_name in ['HAN', 'MHAN']:
        text_sent = [[word2idx[idx] for sub in text[i] for idx in sub] for i in range(cnt)]
        sent_all_att, doc_all_att = att
        normalizer_doc = Normalizer()
        att_sent = normalizer_sent.fit_transform(sent_all_att)
        att_doc = normalizer_doc.fit_transform(doc_all_att)
        customed_heatmap(att_sent, text_sent, n_limit, date, label, 'sent')
        customed_heatmap(att_doc[:,::-1].T, text_sent, n_limit, date, label, 'doc')

    elif model_name == 'Self_Att':
        text_sent = [[word2idx[idx] for idx in text[i]] for i in range(cnt)]
        sent_all_att = att
        att_sent = normalizer_sent.fit_transform(sent_all_att)
        customed_heatmap(att_sent, text_sent, n_limit, date, label, 'sent')


        #important_words = [[word2idx[idx] for idx in word_idx[w_idx]] 
        #                    for w_idx in range(SHOW_SAMPLES_CNT)]
        #print('some important keywords:')
        #pprint(important_words)