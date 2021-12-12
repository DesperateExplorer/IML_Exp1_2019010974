import os
import numpy as np
import pandas as pd


def basic_stat(dataset_dir):
    """
    获取文件大小的最大值以及正负样本的比例
    RETUREN: (文件大小最大值，最大文件的路径，正负样本比例元组)
    """
    df = pd.read_csv(os.path.join(dataset_dir, 'label/index'), delimiter=' ', header=None)
    df[1] = np.array(list(map(lambda item: os.path.join(dataset_dir, item[3:]), df[1])))
    files_sizes = np.array(list(map(lambda file: os.stat(file).st_size, df[1])))
    
    return np.max(files_sizes), df.iloc[np.argmax(files_sizes), 1],  np.unique(df[0], return_counts=True)


if __name__=='__main__':
    extreme_value, ex_path, counts = basic_stat('../../trec06p/')
    print(extreme_value, 'bytes')
    print(ex_path)
    print(counts)

