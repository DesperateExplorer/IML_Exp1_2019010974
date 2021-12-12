import os
import email
import numpy as np
import pandas as pd
from tqdm import tqdm

def stats_in_EDA(dataset_dir, meta_ls, body_ls):
    """
    dataset_dir: 数据集的路径
    meta_ls: 邮件头中关心的符号或串
    body_ls: 邮件正文中关心的符号或串

    将关心的符号和串的统计数据以及“正文中有无html tags”、“邮件是否有乱码”计入指定的`json`文件
    并返回一个字典
        context = {
        'max_file_size': 最大邮件的大小（字节），
        'max_size_file': 最大邮件的路径，
        'dataset_size': 样本量，
        'ham_size': 负样本（阴性）量
        'spam_size': 正（阳性）样本量
    }
    """
    df = pd.read_csv(os.path.join(dataset_dir, 'label/index'), delimiter=' ', header=None)
    df['fullpath'] = [os.path.join(dataset_dir, item[3:]) for item in df[1]]
    res = pd.DataFrame()
    res['label'] = np.array(df[0]) == 'spam'
    res['html'] = np.zeros(len(df[0]), dtype=bool) # 是否有html tags
    res['utf-8'] = np.ones_like(res['html'], dtype=bool) # 是否能被utf-8成功解码打开

    meta_cs = [f'm->\'{me}\'' for me in meta_ls]
    body_cs = [f'b->\'{bo}\'' for bo in body_ls]

    for me in meta_cs:
        res[me] = np.zeros(len(df[0]), dtype=int)
    for bo in body_cs:
        res[bo] = np.zeros(len(df[0]), dtype=int)
    for k, path in tqdm(enumerate(df['fullpath'])):
        try:
            f = open(path)
            msg = email.message_from_file(f)
        except:
            f = open(path, errors='ignore')
            msg = email.message_from_file(f)
            res.loc[k, 'utf-8'] = False
        body = ""
        for part in msg.walk():
            if 'text' in part.get_content_type():
                if 'html' in part.get_content_type():
                    res.loc[k, 'html'] = True
                body += part.get_payload()
        
        header_dict = dict(msg.items())
        header_dict['Received'] = " ".join(msg.get_all('Received'))
        header = " ".join(header_dict.values())
        # for me in meta_ls:
        #     res.loc[k, me] = header.count(me)
        res.loc[k, meta_cs] = [header.count(me) for me in meta_ls]
        # for bo in body_ls:
        #     res.loc[k, bo] = body.count(bo)
        res.loc[k, body_cs] = [body.count(bo) for bo in body_ls]
    
    res['m->#care'] = np.sum(res.loc[:, meta_cs], axis=1)
    res['b->#care'] = np.sum(res.loc[:, body_cs], axis=1)

    res.to_json(f'{os.path.join(dataset_dir, "inter/")}stats_in_EDA.json')
    print(f'数据已存入 ' + f'{os.path.join(dataset_dir, "inter/")}stats_in_EDA.json')

    files_sizes = np.array([os.stat(file).st_size for file in  df['fullpath']])
    context = {
        'max_file_size': np.max(files_sizes),
        'max_size_file': df.iloc[np.argmax(files_sizes), 1],
        'dataset_size': len(df['fullpath']),
        'ham_size': np.unique(df[0], return_counts=True)[1][0],
        'spam_size': np.unique(df[0], return_counts=True)[1][1]
    }
    return context

def stats_to_records(dataset_dir:str, lowercase:bool, info:dict, result:tuple, epoch:int):
    """
    info has fields 'N', 'PERCENTAGE', 'ALPHA', 'CRAFT'
    result is a 4-gram tuple (acc, prec, recall, f1_score)
    epoch: epoch identifier
    """
    s = pd.DataFrame(data={
        'LOWERCASE': [lowercase],
        'N': [info['N']],
        'PERCENTAGE': [info['PERCENTAGE']],
        'ALPHA': [info['ALPHA']],
        'CRAFT': [info['CRAFT']],
        'accuracy': [result[0]],
        'precision': [result[1]],
        'recall': [result[2]],
        'f1_score': [result[3]]
    })
    s.to_csv(os.path.join(dataset_dir, 'inter/records.csv'), mode='a', header=False)
    print(f'Following epoch={epoch} is DONE!\n', s)
