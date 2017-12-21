# -*- coding: utf-8 -*-

'''百度AI情感分析接口调用'''

from aip import AipNlp
import pandas as pd
import re

""" 你的 APPID AK SK """
APP_ID = '10449390 '
API_KEY = 'OKc3wCLxadElQjWGfpvub080'
SECRET_KEY = 'coVGTGNgo3qpTNYC42Cpe6i7Bsld1xkS'
# 创建连接
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

df = pd.read_excel('./data/jd_comp_final_v3.xlsx')
text = df['cleaned_reviews'].tolist()
text = [re.sub(' ','',sent) for sent in text]
result = []
wrong_text = []
for k,sent in enumerate(text):
    if k % 2000 == 0: print(k)
    try:
        res = client.sentimentClassify(text = sent)['items'][0]
        pos = res['positive_prob']
        neg = res['negative_prob']
        label = res['sentiment']
    except:
        print(sent)
        wrong_text.append(sent)
    if pos:
        result.append({'pos':pos, 'neg':neg, 'label':label})
    else:
        result.append({'pos':'', 'neg':'', 'label':''})

sentiment_df = pd.DataFrame(result)

df['pos_prob'] = sentiment_df['pos']
df['neg_prob'] = sentiment_df['neg']
df['sent_label'] = sentiment_df['label']
   
# 将你的文件保存，你可以保存为xlsx文件    
df.to_excel('./data/jd_comp_final_v4.xlsx', index = None)