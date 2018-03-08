# -*- coding: utf-8 -*-
'''百度AI依存句法分析接口调用'''

from aip import AipNlp
import pandas as pd
import re

""" 你的 APPID AK SK """
APP_ID = '10435290'
API_KEY = 'tsxykQc5yyMmBlyUOvIbOwtc'
SECRET_KEY = 'NUXS3Fj7gpGap8Rl32O3zII2SkaHUES3'

# 创建连接
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

df = pd.read_excel('./data/jd_comp_final_v5.xlsx')
text = df[(df['Yes/No'] == 1)&(df['H'] == 0)]['cleaned_reviews'].tolist()
text = [''.join(item.split()) for item in text]

result = []
wrong_text = []

for k,sent in enumerate(text):
    if k % 100 == 0: print(k)
    cut = re.split('。|！|？', sent)
    try:
        for kk,ss in enumerate(cut):
            if ss != '':
                res = client.depParser(ss)['items']
                for item in res:
                    item['sent#'] = str(k)
                    item['sub_sent#'] = str(kk)
                    result.append(item)
            else:
                pass
    except:
        print(k, sent)
        wrong_text.append(sent)

dp_df = pd.DataFrame(result)
dp_df.to_excel('./data/seq_extraction.xlsx', index=None)
