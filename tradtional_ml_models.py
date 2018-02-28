from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
from pprint import pprint

def load_data(data_file):
    df = pd.read_excel(data_file)
    comp = df[df['Yes/No'] == 1]['cleaned_reviews']
    non = df[df['Yes/No'] == 0]['cleaned_reviews']
    hidden = df[df['H'] == 1]['cleaned_reviews']
    not_hidden = df[(df['Yes/No'] == 1) & (df['H'] == 0)]['cleaned_reviews']
    return comp, non, hidden, not_hidden

def get_x_y(dataset):
   x, y = [], []
   for i in range(len(dataset)):
      x = x + dataset[i].tolist()
      y += [i for _ in range(dataset[i].shape[0])]
   return x,y

# has been set best parameters
def tf_idf_model(classifier):
    if classifier == 'svm':
        model = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha = 0.00001, random_state=2017)),
                        ])
    elif classifier == 'nb':
        model = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultinomialNB()),
                        ])

    elif classifier == 'lr':
        model = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                          #('tfidf', TfidfTransformer()),
                          ('clf', LogisticRegression(penalty='l1')),
                        ])
    elif classifier == 'svc':
        model = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
                          ('tfidf', TfidfTransformer()),
                          ('clf', SVC(C=1.0, kernel='rbf')),
                        ])
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

comp, non, hidden, not_hidden = load_data('./data/jd_comp_final_v5.xlsx')
DATASET = [comp, non]
FIND_BEST = False

x, y= get_x_y(DATASET)

model_svm = tf_idf_model('svm')
model_nb = tf_idf_model('nb')
model_lr = tf_idf_model('lr')
model_svc = tf_idf_model('svc')

parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__use_idf': (True, False),
              'clf__C': (1,0.1,0.01,0.001),
              'clf__penalty': ('l1','l2')}
parameters_1 = {'clf__alpha':(0.00001,0.0001,0.001,0.01)}
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2017)

# find best parameters for lr
'''
clf__C: 1
clf__penalty: 'l1'
tfidf__use_idf: True
vect__ngram_range: (1, 2)
'''
if FIND_BEST:
    lr_clf = GridSearchCV(model_svm, parameters_1, n_jobs=-1)
    lr_clf = lr_clf.fit(x,y)
    lr_clf.best_score_
    for param_name in sorted(parameters_1.keys()):
        print("%s: %r" % (param_name, lr_clf.best_params_[param_name]))

result = cv_train(x, y, model_lr)
# res_nb = cv_train(x, y, model_nb) # fail
# res_lr = cv_train(x, y, model_lr)

#pprint(res_lr)
pprint(result)
print(np.mean([row[2] for row in result[0]]))