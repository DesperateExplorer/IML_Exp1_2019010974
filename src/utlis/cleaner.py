import os
import re
import email
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from email.utils import parsedate
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import logging
import pandas as pd
import numpy as np

def clean_globally(dataset_dir ,lowercase=False):
    """
    此函数所做的所有数据预处理全部基于知识，**非数据驱动**，故在**整个**数据集上做；
    此函数**不**包含数据驱动的特征提取

    """
    filename = f'{os.path.join(dataset_dir, "inter/")}processed_items_lowercase={lowercase}.json'
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Cleaning raw data and saving to {filename}')
    df = pd.read_csv(os.path.join(dataset_dir, 'label/index'), delimiter=' ', header=None)
    df['fullpath'] = [os.path.join(dataset_dir, item[3:]) for item in df[1]]
    # processed items
    prsd_items = pd.DataFrame()
    prsd_items['label'] = np.array(df[0]) == 'spam' # True 意为“阳性”，即垃圾邮件

    prsd_items['utf-8'] = np.ones_like(prsd_items['label'], dtype=bool)
    prsd_items['html'] = np.zeros_like(prsd_items['label'], dtype=bool)
    prsd_items['consistent'] = np.ones_like(prsd_items['label'], dtype=bool)
    prsd_items['mess_index'] = np.zeros_like(prsd_items['label'], dtype=int)
    prsd_items['body'] = np.zeros_like(prsd_items['label'], dtype=dict)
    
    meta_ls = ['!', '?', '$']
    for me in meta_ls:
        prsd_items[me] = np.zeros_like(prsd_items['label'], dtype=int)

    prog = re.compile(r"@([\w.]+)")
    for k, path in enumerate(tqdm(df['fullpath'])):
        try:
            f = open(path)
            msg = email.message_from_file(f)
        except:
            f = open(path, errors='ignore')
            msg = email.message_from_file(f)
            prsd_items.loc[k, 'utf-8'] = False
        
        body = str()
        for part in msg.walk():
            if 'text' in part.get_content_type():
                if 'html' in part.get_content_type():
                    prsd_items.loc[k, 'html'] = True
                    temp = part.get_payload()
                    temp = BeautifulSoup(temp, 'lxml').text
                    body += temp
                else:
                    body += part.get_payload()
        rgtkr = RegexpTokenizer(r'([a-zA-Z]+|\!|\$)') # 基于知识，叹号和美元符号也许有判别力
        lmtzr = WordNetLemmatizer()
        body = rgtkr.tokenize(body)
        if lowercase:
            body = [item.lower() for item in body] # 转小写
        # lemmatization
        body = [lmtzr.lemmatize(item) for item in body if len(item) > 3 or item == '!' or item == '$'] # 去除长度不不超过3的词汇
        # 去停用词
        stop_words = set(stopwords.words('english')) 
        body = [item for item in body if item not in stop_words]
        # body = Counter(body)     
        body = ' '.join(body)   
        prsd_items.loc[k, 'body'] = body  
        # 处理consistent
        if msg.get('from') == None or msg.get('message-id') == None:
            prsd_items.loc[k, 'consistent'] = False
        else:
            domain_name = prog.search(msg.get('from'))
            if domain_name == None:
                prsd_items.loc[k, 'consistent'] = False
            elif domain_name.group(1) not in msg.get('message-id'):
                prsd_items.loc[k, 'consistent'] = False
        
        # 若无subject，也说明极可能有问题
        if msg.get('subject') == None:
            prsd_items.loc[k, 'consistent'] = False
        # 原则上date是协议中强制要求的项，并且应该合法
        if parsedate(msg.get('date')) == None:
            prsd_items.loc[k, 'consistent'] = False
        
        ## 处理mess_index（即metadata中问号、叹号、美元符号的数量总和）
        header_dict = dict(msg.items())
        header_dict['Received'] = " ".join(msg.get_all('Received'))
        header = " ".join(header_dict.values())

        prsd_items.loc[k, meta_ls] = [header.count(me) for me in meta_ls]

    prsd_items['mess_index'] = np.sum(prsd_items.loc[:, meta_ls], axis=1)
    prsd_items.drop(meta_ls, axis=1, inplace=True)    
    prsd_items.to_json(filename)