'''百度AI情感分析接口调用'''

from aip import AipNlp
import pandas as pd
import re
from utils import baidu_segment

""" 你的 APPID AK SK """
APP_ID = '10892764'
API_KEY = 'y7Lho7F2NZ8f0oBGzLwzBMrp'
SECRET_KEY = 'YCTODcMvSWREmOLsvF1SBYFZH49WmAe8'

# 创建连接
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

df = pd.read_excel('./data/taobao_20171228.xlsx')
text = df['content'].tolist()

result = []
wrong_text = []

res = baidu_segment(client, text[60000:])
df_s = df.loc[60000:]
df_s['segment'] = res
df_s.to_excel('./data/taobao_v2_60000_90000.xlsx', index = None)