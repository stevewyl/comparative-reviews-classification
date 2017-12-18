from pymongo import MongoClient
import glob
import re
import pandas as pd
import numpy as np
from utils import save_txt_data, read_line_data, clean_text
from utils import segment_sent, check_segment, tra2sim, strQ2B
from utils import baidu_segment
from aip import AipNlp
import jieba

client = MongoClient('mongodb://localhost:27017/')
db = client['jd']
files = glob.glob('./data/labeled_files/*.xlsx')
pattern = re.compile('(?<=\.).*(?=_)')
filelist = [pattern.search(f).group()[1:] for f in files]
filename = sorted(set(filelist), key = filelist.index)

""" 你的 APPID AK SK """
APP_ID = '10435290'
API_KEY = 'tsxykQc5yyMmBlyUOvIbOwtc'
SECRET_KEY = 'NUXS3Fj7gpGap8Rl32O3zII2SkaHUES3'

jieba.load_userdict("./data/jieba_dict.txt")

def remove_duplicate(f):
    xls = pd.ExcelFile(f)
    df = xls.parse(0)
    df['comment_clean'] = df['comment_clean'].fillna('n')
    df = df[df.comment_clean != 'n']
    df = df.reset_index(drop = True)
    df['Yes/No'] = df['Yes/No'].astype(np.float64)
    return df

def change_review(collection, df):
    for index, row in df.iterrows():
        time = row['time']
        user = row['user']
        review_db = collection.find_one({'time':time,'user':user})
        if review_db:
            row['comment_clean'] = review_db['comment']
    return df

def re_clean(df):
    text = df['comment_clean'].tolist()
    cleaned_text = clean_text(text)
    df.loc[:,'reviews'] = cleaned_text
    return df

def cut_word(df, method):
    reviews = df['reviews'].tolist()
    if method == 'jieba':
        cut = segment_sent(reviews, method)
        df.loc[:,'cut_reviews_words'] = cut
    elif method == 'baidu':
        client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
        cut = baidu_segment(client, reviews)
        df.loc[:,'cut_reviews_words'] = cut   
    return df

def cut_char(df):
    reviews = df['reviews'].tolist()
    cut = []
    for s in reviews:
        cut.append(" ".join([st for st in s]))
    df.loc[:,'cut_reviews_char'] = cut
    return df

def df2xlsx(name, df):
    with pd.ExcelWriter(name) as writer:
        df.to_excel(writer,'data', index = False)
        writer.save()

def main_mode_1(files, filelist):
    df_list = []
    for j in range(len(files)):
        print(filelist[j])
        collection = db[filelist[j]]
        df = remove_duplicate(files[j])
        df = change_review(collection, df)
        df['comment_clean'] = df['comment_clean'].apply(tra2sim)
        df['comment_clean'] = df['comment_clean'].apply(strQ2B)
        df = re_clean(df)
        df = cut_word(df, 'baidu')
        df_list.append(df)
    print('reviews clean and segment complete!')
    df_all = pd.concat(df_list)
    df_all = df_all.reset_index(drop = True)
    return df_all
    
def main_mode_2(fname):
    df = pd.read_excel(fname)
    df['comment_clean'] = df['comment_clean'].fillna('n')
    df = df[df.comment_clean != 'n']
    collection_list = list(set(df['product'].tolist()))
    df_list = []
    for name in collection_list:
        print('Dataset Set Name: ', name)
        df_sub = df[df['product'] == name]
        df_sub = change_review(db[name], df_sub)
        df_sub.loc[:,'comment_clean'] = df_sub['comment_clean'].apply(tra2sim)
        df_sub.loc[:,'comment_clean'] = df_sub['comment_clean'].apply(strQ2B)
        df_sub = re_clean(df_sub)
        df_sub = cut_word(df_sub, 'baidu')
        df_list.append(df_sub)
    print('reviews clean and segment complete!')
    df_all = pd.concat(df_list)
    df_all = df_all.reset_index(drop = True)
    return df_all

if __name__ == '__main__':
    df_all = main_mode_1(files, filelist)
    df_all.to_excel('./data/jd_comp_final.xlsx', index = False)
    #df_all = main_mode_2(files, filelist)
    columns = ['Yes/No', 'H', '平比', '差比', 'cut_reviews_words',
               '比较主体', '比较客体', '比较点', '比较结果', '比较标记']
    df_all = df_all[columns]
    df_all['Yes/No'] = df_all['Yes/No'].astype(object)
    df_all['H'] = df_all['H'].astype(object)
    df_all = df_all.fillna(0)
    df_all.to_pickle('./data/problem1_data_03.pkl')
    df_comp_word = df_all.loc[df_all['Yes/No'] == 1, 'cut_reviews_words']
    df_non_comp_word = df_all.loc[df_all['Yes/No'] == 0, 'cut_reviews_words']
    df_not_hidden_word = df_all.loc[(df_all['Yes/No'] == 1) & (df_all['H'] == 0), 'cut_reviews_words']
    df_hidden_word = df_all.loc[(df_all['Yes/No'] == 1) & (df_all['H'] == 1), 'cut_reviews_words']
    #df_comp_char = df_all.loc[df_all['Yes/No'] == 1.0, 'cut_reviews_char']
    #df_non_comp_char = df_all.loc[df_all['Yes/No'] == 0.0, 'cut_reviews_char']
    filepath = 'C:/Users/stevewyl/Desktop/paper_scripts/data/'
    df_comp_word.to_csv(filepath + 'comp_reviews_word_03.csv', index = None, encoding = 'utf-8')
    df_non_comp_word.to_csv(filepath + 'non_comp_reviews_word_03.csv', index = None, encoding = 'utf-8')
    df_not_hidden_word.to_csv(filepath + 'not_hidden_reviews_word_03.csv', index = None, encoding = 'utf-8')
    df_hidden_word.to_csv(filepath + 'hidden_reviews_word_03.csv', index = None, encoding = 'utf-8')
    #df_comp_char.to_csv(filepath + 'comp_reviews_char.csv', index = None, encoding = 'utf-8')
    #df_non_comp_char.to_csv(filepath + 'non_comp_reviews_char.csv', index = None, encoding = 'utf-8')
    