# -*- coding: utf-8 -*-

'''百度AI分词接口调用'''

from aip import AipNlp
# 这里是我调用的自己写的函数，具体就是一行行地把数据读进来
from utils import read_line_data, save_txt_data
import re

""" 你的 APPID AK SK """
APP_ID = '10449390 '
API_KEY = 'OKc3wCLxadElQjWGfpvub080'
SECRET_KEY = 'coVGTGNgo3qpTNYC42Cpe6i7Bsld1xkS'
# 创建连接
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

cleaned_text = read_line_data('./data/cleaned_text.txt', False)
cleaned_text  = [re.sub(' ', '', sent) for sent in cleaned_text]
all_text = []
wrong_text = []
print('start segment')
for k,sent in enumerate(cleaned_text[120000:170000]):
    if k % 2000 == 0: print(k)
    try:
        res = AipNlp.lexer(client, text = sent) # 改为情感分析的接口
        seg_text = [word['item'] for word in res['items']] # 根据你返回的结果，字典的key可能不一样
        all_text.append(' '.join(seg_text))
    except:
        print(sent)
        wrong_text.append(sent)
   
# 将你的文件保存，你可以保存为xlsx文件    
save_txt_data('./data/baidu_segemented_text_120000_170000.txt', all_text, 'text')