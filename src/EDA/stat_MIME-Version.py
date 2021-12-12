import pandas as pd
from tqdm import tqdm
import os

PATTERN = 'mime-version: 1.0' # all plain text are converted to lower in this script


def get_files():

    df = pd.read_csv('../../trec06p/label/index', delimiter=' ', header=None)
    df[1] = [os.path.join('../../trec06p/', item[3:]) for item in df[1]] 
    for path in df[1]:        
        f = open(path, errors='ignore')
        temp = f.readlines()
        f.close()
        
        temp = [s.strip().lower() for s in temp]
        yield temp

def get_TF(content, pattern: str):

    return True if pattern in content else False

if __name__ == '__main__':
    have = [get_TF(cont, PATTERN) for cont in tqdm(get_files())]
    s = pd.Series(data=have, name=PATTERN)
    s.to_csv('../../MIME-Version.csv')