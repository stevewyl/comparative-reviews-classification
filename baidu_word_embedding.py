from aip import AipNlp
from utils import read_line_data, save_txt_data
from keras.preprocessing.text import Tokenizer
import pandas as pd

df = pd.read_excel('./data/final_data.xlsx')
reviews = df['cut_reviews'].tolist()

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower = True, split = " ")
tokenizer.fit_on_texts(reviews)
vocab = tokenizer.word_index

words = list(vocab.keys())
word_cnt = len(words)
n_dim = 1024

APP_ID_1 = '10435290'
API_KEY_1 = 'tsxykQc5yyMmBlyUOvIbOwtc'
SECRET_KEY_1 = 'NUXS3Fj7gpGap8Rl32O3zII2SkaHUES3'
client1 = AipNlp(APP_ID_1, API_KEY_1, SECRET_KEY_1)

APP_ID_2 = '10440162'
API_KEY_2 = 'LsK8vpUUPGTeXzacjYKQkyC6'
SECRET_KEY_2 = 'lqW3KDu7pPP4zxWvBgZR3uhmZoCNS9mv'
client2 = AipNlp(APP_ID_2, API_KEY_2, SECRET_KEY_2)

APP_ID_3 = '10449390'
API_KEY_3 = 'OKc3wCLxadElQjWGfpvub080'
SECRET_KEY_3 = 'coVGTGNgo3qpTNYC42Cpe6i7Bsld1xkS'
client3 = AipNlp(APP_ID_3, API_KEY_3, SECRET_KEY_3)

word_embeddings = {}
for k,word in enumerate(words):
    if k < 8000:
        client = client1
    elif k > 8000 and k < 16000:
        client = client2
    else:
        client = client3
    try:
        result = client.wordEmbedding(word)
        word_embeddings[word] = result['vec']
    except:
        word_embeddings[word] = [0] * n_dim
    if k % 1000 == 0: 
        print(k)

fname = './data/baidu_1024d_1205.txt'
with open(fname, 'w', encoding = 'utf-8') as file:
    file.write(str(word_cnt) + ' ' + str(n_dim))
    file.write('\n')
    for k,v in word_embeddings.items():
        v = [str(i) for i in v]
        line = k + ' ' + ' '.join(v)
        file.write(line)
        file.write('\n')
    file.close()