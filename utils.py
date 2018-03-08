from pymongo import MongoClient
import pandas as pd
import numpy as np
import re
import jieba
# import thulac
from nltk.corpus import wordnet
# from wordsegment import segment
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import StratifiedKFold
from langconv import Converter

from aip import AipNlp
from flashtext import KeywordProcessor
import math

# 连接数据库获取所有文本语料
def connect_mongo():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['jd']
    product_list = [name for name in db.collection_names() 
                    if not name.startswith('links') and not name.startswith('page')]
    return db, product_list

# 读取所有数据
def read_data(product_list, db):
    df_all = []
    for name in product_list:
        collection = db[name]
        df = [item for item in collection.find()]
        df = pd.DataFrame(df)
        df['product_name'] = [name] * df.shape[0]
        df_all.append(df)
    return pd.concat(df_all)

# 繁转简（已解决）
def tra2sim(text):
    return Converter('zh-hans').convert(text)

# 全角转半角
def strQ2B(sent):
    rstring = ""  
    for uchar in sent:  
        inside_code = ord(uchar)  
        if inside_code == 12288:    #全角空格直接转换              
            inside_code = 32  
        elif inside_code >= 65281 and inside_code <= 65374: #全角字符（除空格）根据关系转化  
            inside_code -= 65248  
            rstring += chr(inside_code)
        else:
            rstring += chr(inside_code)
    return rstring

# 保存为txt文件
def save_txt_data(filename, text, mode):
    file = open(filename, 'w', encoding = 'utf-8')

    for sent in text:
        file.write(sent)
        file.write('\n')
    file.close()
    wrong_len = [len(sent.split()) for sent in text]
    print('avg_len wrong: ', np.mean(wrong_len))
    '''
    elif mode == 'seq':
        new_text = [re.sub('UNK', '', ' '.join(seq)) for seq in text]
        for sent in new_text:
            file.write(sent)
            file.write('\n')
        file.close()
        wrong_len = [len(sent.split()) for sent in new_text]
        print('avg_len wrong: ', np.mean(wrong_len))
    '''

# 按行读入评论
def read_line_data(filename, is_dict = False):
    file = open(filename, encoding='utf-8')
    if not is_dict:
        return [line.strip() for line in file.readlines()]
    else:
        wrong = pd.read_csv(filename)
        wrong = wrong.set_index('false')
        wrong_dict = wrong['true'].to_dict()
        return wrong_dict

# 正则表达式清理文本
def clean_text(text):
    # 需要去除的标点符号
    punctuation_remove = '[�｀┐◎╯┘‖＃┻╮╰＂╭●─●﹏〃︵ノ﹋ヾへ＆づ\≧≦※+⊙⊙★☆＝▽↖↗：；·（）『』《》【】～!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]+'
    need_remove = '[⑧⑩⑤⑦┉◇▼▄灬╥╬〔┝ˊ＼￢／ｖレ丯△ナ]〞+'
    new = []
    p = re.compile(r'\w+|[\u4e00-\u9fa5]+|[，。.！？]+')
    for sent in text:
        sent = sent.lower()
        # 一些标点符号的转换
        sent = re.sub(' ', '，', sent)
        sent = re.sub(',', '，', sent)
        sent = re.sub('丶', '', sent)
        sent = re.sub(r'\\', '', sent)
        sent = re.sub(r'(?<!\d)\.', '。', sent)
        sent = re.sub(r'\?', '？', sent)
        sent = re.sub(r'!', '！', sent)
        sent = re.sub('\d、', '。', sent)
        sent = re.sub('&\w*;', '', sent)
        sent = re.sub(r'\n', '', sent)
        # 只保留英文字母数字和中文字和中文的标点符号
        sent = p.findall(sent)
        sent = ''.join(sent)
        # 一些过长词的处理
        sent = re.sub('[a-zA-Z]{10,}', ' ', sent)
        sent = re.sub(r'(\d){6,}', '', sent)
        sent = re.sub(r'6{4,}', '666', sent)
        sent = re.sub(r'h{4,}', 'hhh', sent)
        sent = re.sub(r'z{4,}', 'zzz', sent)
        sent = re.sub(r'买{4,}', '买买买', sent)
        sent = re.sub(r'贵{4,}', '贵贵贵', sent)
        sent = re.sub(r'六{4,}', '六六六', sent)
        sent = re.sub('vv', '', sent)
        sent = re.sub('void0', '', sent)
        sent = re.sub(r'(x|湛|差|哦|了|咣|棒|十五字|爽|耶|赞|啦|哈|啊|豪|恩|嗯|不错|好|很好|可以|还好|差评){5,}', '', sent)
        # 错误的品牌名及术语修正
        sent = re.sub(r'viv0', 'vivo', sent)
        sent = re.sub('36o', '360', sent)
        sent = re.sub(r'维沃', 'vivo', sent)
        sent = re.sub('三t', '3t', sent)
        sent = re.sub('0ppo|opp0|0pp0|opop', 'oppo', sent)
        sent = re.sub(r'opp(?!o)', 'oppo', sent)
        sent = re.sub(r'waif|wi-fi|wif|wfi|wiff|wife|wifl|wify|wiifi|wifve', 'wifi', sent)
        sent = re.sub('wifii', 'wifi', sent)
        sent = re.sub('puls|pius|pluse', 'plus', sent)
        sent = re.sub(r'(\d)pus', '\\1plus', sent)
        sent = re.sub(r'(?<=\d)p(?!lus|ro|\d)', 'plus', sent)
        sent = re.sub(r'(?<=\ds)p(?!lus|ro|\d)', 'plus', sent)
        sent = re.sub(r'(720|1080)plus', '\\1p', sent)       
        sent = re.sub(r'(?<=0)w', '万', sent)
        sent = re.sub(r'(?<=0)ma', ' mah', sent)
        sent = re.sub('usf', 'ufs', sent)
        sent = re.sub(r'not(?=\d)', 'note', sent)
        sent = re.sub('n0te|noet|noto|n0t', 'note', sent)
        sent = re.sub(r'ip(?=\d)', 'iphone', sent)
        sent = re.sub('ihone', 'iphone', sent)
        sent = re.sub('爱疯', 'iphone', sent)
        sent = re.sub('某宁', '苏宁', sent)
        sent = re.sub('某东', '京东', sent)
        sent = re.sub('某猫', '天猫', sent)
        sent = re.sub('某宝|某bao', '淘宝', sent)
        sent = re.sub(r'((?<=[\u4e00-\u9fa5]))por', '\\1pro', sent)
        # 拼音修正
        sent = re.sub('keyi', '可以', sent)
        sent = re.sub('qiang', '强', sent)
        sent = re.sub('nima', '尼玛', sent)
        sent = re.sub('kendie|坑die|坑d', '坑爹', sent)
        sent = re.sub('keng', '坑', sent)
        sent = re.sub('laji', '垃圾', sent)
        sent = re.sub('henhao', '很好', sent)
        sent = re.sub('jingdong', '京东', sent)
        sent = re.sub('bucuo', '不错', sent)
        sent = re.sub('二shou|2shou', '二手', sent)
        sent = re.sub('huawei', '华为', sent)
        sent = re.sub('zhifubao|zfb', '支付宝', sent)
        sent = re.sub('ri', '日', sent)
        # 中文错别字纠正
        sent = re.sub('hone键', 'home健', sent)
        sent = re.sub('馀', '余', sent)
        sent = re.sub('信懒', '信赖', sent)
        sent = re.sub('可口可乐', '', sent)
        sent = re.sub('像数|相素', '像素', sent)
        sent = re.sub('冲电', '充电', sent)
        sent = re.sub('快冲', '快充', sent)
        sent = re.sub('炒鸡', '超级', sent)
        sent = re.sub('反修', '返修', sent)
        sent = re.sub('保护模(?!式)', '保护膜', sent)
        sent = re.sub('(优惠|京|满减)卷', '\\1券', sent)
        sent = re.sub('晓龙', '骁龙', sent)
        sent = re.sub('牛比', '牛逼', sent)
        sent = re.sub('萍果|平果', '苹果', sent)
        sent = re.sub('home健', 'home键', sent)
        sent = re.sub('为毛', '为什么', sent)
        # 强行加空格帮助分词
        '''
        sent = re.sub(r'我(?!们)', '我 ', sent)
        sent = re.sub('很', '很 ', sent)
        sent = re.sub('太', '太 ', sent)
        sent = re.sub('根本', '根本 ', sent)
        sent = re.sub('接听', '接听 ', sent)
        sent = re.sub('比较', '比较 ', sent)
        sent = re.sub('无论', '无论 ', sent)
        sent = re.sub('经常', '经常 ', sent)
        sent = re.sub('功能', '功能 ', sent)
        sent = re.sub('速度', '速度 ', sent)
        sent = re.sub('块钱', '块 钱', sent)
        sent = re.sub('手机', ' 手机', sent)
        sent = re.sub('方便快捷', '方便 快捷', sent)
        sent = re.sub('十分满意', '十分 满意', sent)
        sent = re.sub('就行了', '就 行了', sent)
        sent = re.sub('使用方便', '使用 方便', sent)
        sent = re.sub('不是太好', '不是 太 好', sent)
        sent = re.sub('拿在手上', '拿 在 手上', sent)
        sent = re.sub('真不知道', '真 不 知道', sent)
        sent = re.sub('正合适', '正 合适', sent)
        sent = re.sub(r'(昨天|今天|前天)', '\\1 ', sent)
        sent = re.sub('过段时间', '过 段 时间', sent)
        sent = re.sub('价格', '价格 ', sent)
        sent = re.sub('反应(迟钝|速度)', '反应 \\1', sent)
        sent = re.sub('挺', '挺 ', sent)
        sent = re.sub('非常', '非常 ', sent)
        sent = re.sub('灰常', '非常 ', sent)
        sent = re.sub('弓虽', '强', sent)
        sent = re.sub('土亢', '坑', sent)
        sent = re.sub('是否是', '是否 是', sent)
        sent = re.sub('好开心', '好 开心', sent)
        sent = re.sub('内存不足', '内存 不足', sent)
        sent = re.sub('屏太', '屏 太', sent)
        sent = re.sub('买时', '买 时', sent)
        sent = re.sub('不敢相信', '不敢 相信', sent)
        sent = re.sub('地好', '地 好', sent)
        sent = re.sub('强烈要求', '强烈 要求', sent)
        sent = re.sub('就别', '就 别', sent)
        sent = re.sub('要冲', '要 充', sent)
        sent = re.sub('但用', '但 用', sent)
        sent = re.sub('感觉良好', '感觉 良好', sent)
        sent = re.sub('这么久', '这么 久', sent)
        sent = re.sub('是因为', '是 因为', sent)
        sent = re.sub('看电视', '看 电视', sent)
        sent = re.sub('千万别', '千万 别', sent)
        sent = re.sub('再也不会', '再 也 不会', sent)
        sent = re.sub('解决问题', '解决 问题', sent)
        sent = re.sub('真是太', '真是 太', sent)
        sent = re.sub('机了', '机 了', sent)
        sent = re.sub('却是', '却 是', sent)
        sent = re.sub('才行', '才 行', sent)
        sent = re.sub('能撑', '能 撑', sent)
        sent = re.sub('发现', '发现 ', sent)
        sent = re.sub('拍出来', '拍 出来', sent)
        sent = re.sub('实在太', '实在 太', sent)
        sent = re.sub('从来没', '从来 没', sent)
        sent = re.sub('特别是在', '特别 是 在', sent)
        sent = re.sub(r'其他(?!人)', '其他 ', sent)
        sent = re.sub(r'特别(?!版)', '特别 ', sent)
        sent = re.sub('多少', '多少 ', sent)
        sent = re.sub('设置成', '设置 成', sent)
        sent = re.sub('百分之', '百分之 ', sent)
        sent = re.sub('都还不', '都 还 不', sent)
        sent = re.sub('爸用', '爸 用', sent)
        sent = re.sub('妈用', '妈 用', sent)
        sent = re.sub('就裂', '就 裂', sent)
        sent = re.sub('星是', '星 是', sent)
        sent = re.sub('日买', '日 买', sent)
        sent = re.sub('人理', '人 理', sent)
        sent = re.sub('充不', '充 不', sent)
        sent = re.sub('爸买', '爸 买', sent)
        sent = re.sub('一冲', '一充', sent)
        sent = re.sub('万多', '万 多', sent)
        sent = re.sub('视屏', '视频', sent)
        sent = re.sub('手后', '手 后', sent)
        sent = re.sub('安桌', '安卓', sent)
        sent = re.sub('货是', '货 是', sent)
        sent = re.sub('卡到', '卡 到', sent)
        sent = re.sub('中(要|好)', '中 \\1', sent)
        sent = re.sub('没太', '没 太', sent)
        sent = re.sub('点(前|后|半)', '点 \\1', sent)
        sent = re.sub('这破', '这 破', sent)
        sent = re.sub('先(给|说)', '先 \\1', sent)
        sent = re.sub('天 无', '天 无', sent)
        sent = re.sub('个赞', '个 赞', sent)
        sent = re.sub('看个', '看 个', sent)
        sent = re.sub('差太', '差 太', sent)
        sent = re.sub('来评', '来 评', sent)
        sent = re.sub('电用', '电 用', sent)
        sent = re.sub('是从', '是 从', sent)
        sent = re.sub('人用', '人 用', sent)
        sent = re.sub('货已', '货 已', sent)
        sent = re.sub('时买', '时 买', sent)
        '''
        #去除过多的！，？，。等
        sent = re.sub(r'(，){2,}', '，', sent)
        sent = re.sub(r'(！){2,}', '！', sent)
        sent = re.sub(r'(？){2,}', '？', sent)
        sent = re.sub(r'(。){2,}', '。', sent)
        sent = re.sub(r'(、){2,}', '', sent)
        sent = re.sub(r'(。，)|(，。)', '。', sent)
        sent = re.sub(r'(？，)|(，？)', '？', sent)
        sent = re.sub(r'(！，)|(，！)', '！', sent)
        # 其他
        sent = re.sub('0k', 'ok', sent)
        sent = re.sub('高好', '', sent)
        sent = re.sub('八八八八八|八八', '', sent)
        sent = re.sub('- -', '', sent)
        sent = re.sub(r'|||- -', '', sent)
        sent = re.sub('13t', '一加3t', sent)
        sent = sent.strip()
        if len(sent) > 5:
            new.append(sent)
        else:
            new.append('')
    return new

# 正则切分英文字母和数字（暂弃）
def split_num_char(word):
    special_word_list = read_line_data('./data/keywords_en.txt')
    word_list = re.findall(r'[0-9]+|[a-z]+', word)
    new = []
    for word in word_list:
        if not word.isdigit():
            for w in segment(word):
                if w in special_word_list or wordnet.synsets(w):
                    new.append(w)
        else:
            new.append(word)
    return new

# 去除胡乱输入的乱码（暂弃）
def remove_nonsense(text):
    if ord(text[-1]) in range(97,122) or ord(text[-1]) in range(48,58):
        if not wordnet.synsets(text):
            if len(text) < 10:
                text = " ".join(split_num_char(text))
                text = re.sub(r'(\d) ([gdsk])', '\\1\\2', text)
                text = re.sub(r'([vx]) (\d)', '\\1\\2', text)
                text = re.sub(r'n it', 'nit', text)
                text = re.sub('x play', 'xplay', text)
                text = re.sub('i pad', 'ipad', text)
                text = re.sub('i phone', 'iphone', text)
                text = re.sub(r'(\d) plus', '\\1p', text)
                text = re.sub('fps', 'fps ', text)
            else:
                text = ' '
    return text

# 分词后会出现一些错误，再次纠正
def second_re(sent):
    sent = re.sub(r'or9', 'r9', sent)
    sent = re.sub(r'2.5 d', '2.5d', sent)
    sent = re.sub(r'g b', 'gb', sent)
    sent = re.sub(r'type - c', 'type-c', sent)
    sent = re.sub(r'(\d) . (\d)g', '\\1.\\2g', sent)
    sent = re.sub(r'app le', 'apple', sent)
    return sent
    
# 分词
def segment_sent(text, method = 'jieba'):
    jieba.load_userdict("./data/jieba_dict.txt") 
    thu = thulac.thulac(seg_only = True)
    new = []
    if method == 'jieba':
        for sent in text:
            new_sent = []
            for word in jieba.cut(sent):
                if word != ' ' and len(word) < 10:
                    new_sent.append(word)
            if len(new_sent) > 2:
                new_sent = second_re(" ".join(new_sent))                
                new.append(new_sent)
    elif method == 'thulac':
        for sent in text:
            new_sent = thu.cut(sent, text = True)
            new_sent = re.sub(r'( ){2,}', ' ', new_sent)
            new.append(new_sent)
    return new

# 调用百度api进行分词
def baidu_segment(client, text):
    result = []
    for k,sent in enumerate(text):
        if k % 2000 == 0: print(k)
        if type(sent) == str:
            try:
                sent = re.sub(' ', '', sent)
                res = client.lexer(sent)
                seg_text = [word['item'] for word in res['items']]
                seg_text = [word for word in seg_text if word != ' ' and len(word) < 10]
                if len(seg_text) > 2:
                    result.append(' '.join(seg_text))
                else:
                    result.append('')
            except:
                print(k, sent)
                new_sent = ' '.join(jieba.cut(sent))
                if len(new_sent) > 2:
                    result.append(new_sent)
                else:
                    result.append('')
        else:
            result.append('')
    return result

# 查看分词情况
def check_segment(text):
    a = Counter()
    for sent in text:
        for word in sent.split():
            a[word] += 1
    a = dict(a)
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(text)
    word = vectorizer.get_feature_names()
    word = pd.DataFrame(word)
    word.columns = ['word']
    word['len']=word['word'].apply(len)
    word['count'] = word['word'].map(a)
    word = word.sort_values(['len','count'], ascending=False)
    return word, a



# 预测结果的array数据类型转list 
def array2list(arr):
    return [k for row in arr for k,v in enumerate(row) if v == 1]

# 统计句子中的标点符号
# 1. 无标点-->固定长度切分
# 2. 句号或问号或感叹号数量超过10个--->相邻两个句子合并为一个句子
# 3. 逗号数量超过15个--->相邻三个短句合并为一个新的短句
# 4. 正常的句子--->按照句号等切分
# 5. 某一分句过长--->按照长度进行切分
def check_punc(sent):
    n_comma = sent.count('，')
    n_stop = sent.count('。')
    n_question = sent.count('？')
    n_exc = sent.count('！')
    total = n_comma + n_stop + n_question + n_exc

    if total == 0:
        return 0
    elif n_stop > 10 or n_question > 10 or n_exc > 10:
        return 1
    elif n_comma > 15:
        return 2
    else:
        return 3
    
def collect_sent_len(x):
    res = []
    for sent in x:
        n_comma = sent.count('，')
        n_stop = sent.count('。')
        n_question = sent.count('？')
        n_exc = sent.count('！')
        total = n_comma + n_stop + n_question + n_exc
        if total == 0:
            res.append(5)
        elif n_stop > 10 or n_question > 10 or n_exc > 10:
            res.append(int(total-n_comma)/2)
        elif n_comma > 15:
            res.append(int(n_comma /3))
        else:
            res.append(len(re.split('。|！|？',sent)))
    return res            
# 补齐或截断文本
def standardize_split_sent(splited, interval, n_sents):
    if len(splited) <= n_sents:
        for i in range(n_sents - len(splited)):
            unknown = ['UNK'] * interval
            splited.append(' '.join(unknown))
    else:
        splited = splited[0:n_sents]
    return splited

# 按照具体的分句长度进行切分评论
def cut_sent_by_punc(punc_split, interval):
    splited = []
    len_list = [len(each.split()) for each in punc_split]
    tmp = 0
    start = 0
    for i in range(len(punc_split)):
        tmp += len_list[i]
        if tmp > interval:
            if start == i:
                splited.append(punc_split[start:i+1])
                start = i + 1
                tmp = 0
            else:
                splited.append(punc_split[start:i])
                if len_list[i] > interval:
                    splited.append(punc_split[i:i+1])
                    start = i + 1
                    tmp = 0
                else:
                    start = i
                    tmp = len_list[i]
        if i == len(punc_split) - 1:
            splited.append(punc_split[start:i+1])
    splited = [' '.join(sent) for sent in splited]
    splited[-1] = splited[-1][0:-2]
    return splited

# 按一定规则切分句子
def split_sent(sent, interval, n_sents, mode):
    sent_split = sent.split()
    sent_sub = re.sub(r'(。|！|？)', '\\1 cut ', sent)
    if mode == 'simple':
        splited = [sent_split[i:i+interval] for i in range(0, len(sent_split), interval)]
        splited = [' '.join(sent) for sent in splited]
        return standardize_split_sent(splited, interval, n_sents)
    elif mode == 'complicated':
        res = check_punc(sent)
        if res == 0:
            splited = [sent_split[i:i+interval] for i in range(0, len(sent_split), interval)]
            splited = [' '.join(sent) for sent in splited]
            return standardize_split_sent(splited, interval, n_sents)
        elif res == 1:
            punc_split = re.split(' cut ', sent_sub)
            splited = cut_sent_by_punc(punc_split, interval)
            return standardize_split_sent(splited, interval, n_sents)
        elif res == 2:
            sent_sub = re.sub(r'(，)', '\\1 cut ', sent)
            punc_split = re.split(' cut ', sent_sub)
            splited = cut_sent_by_punc(punc_split, interval)
            return standardize_split_sent(splited, interval, n_sents)
        else:
            punc_split = re.split(' cut ', sent_sub)
            sent_len = [len(each.split()) for each in punc_split]
            if len(sent_len) == 1:
                if sent_len[0] < interval + 5:
                    return standardize_split_sent(punc_split, interval, n_sents)
                else:
                    sent_sub = re.sub(r'(，)', '\\1 cut ', sent)
                    comma_split = re.split(' cut ', sent_sub)
                    splited = cut_sent_by_punc(comma_split, interval)
                    return standardize_split_sent(splited, interval, n_sents)
            else:
                splited = cut_sent_by_punc(punc_split, interval)
                return standardize_split_sent(splited, interval, n_sents)

def shuffle_split_datasets(X, y):
    skf = StratifiedKFold(n_splits = 5, random_state = 2017, shuffle = True)
    skf.get_n_splits(X, y)
    idx_list = [[train_index, test_index] for train_index, test_index in skf.split(X, y)]
    X_train = [X[idx] for idx in idx_list[0][0]]
    X_test = [X[idx] for idx in idx_list[0][1]]
    y_train = [y[idx] for idx in idx_list[0][0]]
    y_test = [y[idx] for idx in idx_list[0][1]]
    test_index = idx_list[0][1]
    return X_train, X_test, y_train, y_test, test_index

# 修正分词错误
def fix_segment(segmented_text):
    keyword_processor = KeywordProcessor()
    fix_word = pd.read_csv('./data/baidu_wrong_segment.csv')
    fix_word = fix_word.fillna('')
    wrong = fix_word['wrong'].tolist()
    correct = fix_word['correct'].tolist()
    for i in range(fix_word.shape[0]):    
        keyword_processor.add_keyword(wrong[i], correct[i])
    segmented_text = [keyword_processor.replace_keywords(sent) for sent in segmented_text]
    return segmented_text

# 根据字典抽取关键字
def extract_keyword(keyword_list, text):
    keyword_processor = KeywordProcessor()
    text = [re.sub(' ','',p) for p in text]
    res = []
    for word in keyword_list:
        keyword_processor.add_keyword(word)
    for phrase in text:
        keywords_found = keyword_processor.extract_keywords(phrase.lower())
        keywords_found = list(set(keywords_found))
        if len(keywords_found) > 0:
            res.append(keywords_found[0])
        else:
            res.append('')
    return res

# 列数据分箱操作
def binning(col, cut_points, labels=None):
    minval = col.min()
    maxval = col.max()
    break_points = [minval] + cut_points + [maxval]
    if not labels:
        labels = range(len(cut_points)+1)
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
    return colBin

def check_data_info(df):
    not_hidden_comp_text = df[(df['Yes/No'] == 1) & (df['H'] == 0)]['cleaned_reviews'].apply(lambda x: len(x.split())).tolist()
    non_comp_text = df[df['Yes/No'] == 0]['cleaned_reviews'].apply(lambda x: len(x.split())).tolist()
    comp_text= [df[(df['差比'] == 1) & (df['H'] == 0)]['cleaned_reviews'].apply(lambda x: len(x.split())).tolist(),
                      df[(df['差比'] == 2) & (df['H'] == 0)]['cleaned_reviews'].apply(lambda x: len(x.split())).tolist(),
                      df[(df['差比'] == 3) & (df['H'] == 0)]['cleaned_reviews'].apply(lambda x: len(x.split())).tolist()]
    equal_text = df[(df['平比'] == 1) & (df['H'] == 0)]['cleaned_reviews'].apply(lambda x: len(x.split())).tolist()
    hidden_text = df[df['H'] == 1]['cleaned_reviews'].apply(lambda x: len(x.split())).tolist()
    all_text = df['cleaned_reviews'].apply(lambda x: len(x.split())).tolist()
 
    print('review length details:\n')
    print('所有评论平均词数: ', int(np.mean(all_text)))
    print('非隐性比较句: ', int(np.mean(not_hidden_comp_text)))
    print('非比较句: ', int(np.mean(non_comp_text)))
    print('平比句: ', int(np.mean(equal_text)))
    print('差比句: ', int(np.mean(comp_text[0])))
    print('不同句: ', int(np.mean(comp_text[1])))
    print('极比句: ', int(np.mean(comp_text[2])))
    print('隐性比较句: ', int(np.mean(hidden_text)))
    print('最长最短隐性比较句: ', int(np.max(hidden_text)), int(np.min(hidden_text)))
    print('最长最短非隐性比较句: ', int(np.max(not_hidden_comp_text)), int(np.min(not_hidden_comp_text)))

    
def check_product_cnt(x, p_l):
    occur_list = []
    x_seg = x['cleaned_reviews'].split()
    entity = x['comp_entity']
    for p in p_l:
        if p in x_seg and p != entity:
            occur_list.append(p)
    occur_list = list(set(occur_list))
    if len(occur_list) >= 1:
        return occur_list, len(occur_list)
    else:
        return occur_list, 0
    