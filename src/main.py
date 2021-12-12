import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from argparse import ArgumentParser, Namespace
from sklearn.metrics import precision_recall_fscore_support

from utlis.cleaner import clean_globally
from utlis.stats import stats_to_records
from models.nbc import NBC
# 5-fold cross validation
K = 5 

# determine whether transform all body words into lowercase
LOWERCASE = {
    'default': False,
    'grid': np.array([
        False, True
    ])
}
# expected number of bag-of-words features
N = {
    'default': 15,
    # The default value of N=15 is from [Better Bayesian Filtering](http://www.paulgraham.com/better.html)
    'grid': np.logspace(start=0, stop=3, num=4, dtype=int)
}
# the size of the training set
PERCENTAGE = {
    'default': 1,
    'grid': np.array([
        0.05, 0.25, 0.5, 1
    ])
}
# smoothing factor for laplace smoothing
ALPHA = {
    'default': 1.0,
    'grid': np.logspace(-2, 4, num=7)
}
# whether hand-crafted features are used
CRAFT = {
    'default': True,
    'grid': np.array([
        False, True
    ])
}

def cv(model: NBC, data: pd.DataFrame, show_words=False):
    """
    K(=5)-Fold Cross Validation
    return (accuracy, precision, recall, f1_score)
    """
    sample_size = data.shape[0]
    data = data.sample(frac=1, random_state=2019010974)
    test_size = int(sample_size/K)
    accs = np.zeros(K, dtype=float)
    precisions = np.zeros(K, dtype=float)
    recalls = np.zeros(K, dtype=float)
    f1s = np.zeros(K, dtype=float)

    for k in range(K):
        ind = np.zeros(sample_size, dtype=bool)
        ind[k*test_size:(k+1)*test_size] = True
        test = data.loc[ind, :].copy()
        train = data.loc[~ind, :].copy()
        model.fit(train)

        if show_words:
            print(model.informative_words())

        ground_truth = test['label'].copy()
        test.drop('label', axis=1, inplace=True)
        pred = model.predict(test)
        acc = pred == ground_truth
        acc = np.sum(acc)/len(acc)
        accs[k] = acc
        precisions[k], recalls[k], f1s[k] =  precision_recall_fscore_support(ground_truth, pred, average='binary')[:3]  
 
    return np.mean((accs, precisions, recalls, f1s), axis=1)
    
if __name__ == '__main__':

    # 1. 生成经全局预处理的数据供之后调用
    for lcs in LOWERCASE['grid']:
        clean_globally('../trec06p/', lowercase=lcs)
    print('Preprocessing is DONE!')

    df = dict()
    for i in LOWERCASE['grid']:
        df[i] = pd.read_json(f"../trec06p/inter/processed_items_lowercase={i}.json")

    epoch = 0

    # 2. 首先跑一个默认参数的实验
    model = NBC(
        alpha=ALPHA['default'],
        percentage=PERCENTAGE['default'],
        hand_crafted=CRAFT['default'],
        N = N['default']
        )
    stats_to_records(
        dataset_dir='../trec06p/',
        lowercase=LOWERCASE['default'],
        info=model.info(),
        result=cv(model=model, data=df[LOWERCASE['default']], show_words=True),
        epoch=epoch
    )
    epoch += 1

    # 3. grid search :)
    for lowercase in LOWERCASE['grid']:
        for n in N['grid']:
            for percentage in PERCENTAGE['grid']:
                for alpha in ALPHA['grid']:
                    for craft in CRAFT['grid']:
                        model = NBC(
                            alpha=alpha,
                            percentage=percentage,
                            hand_crafted=craft,
                            N=n
                        )

                        stats_to_records(
                            dataset_dir='../trec06p/',
                            lowercase=lowercase,
                            info=model.info(),
                            result=cv(model=model, data=df[lowercase]),
                            epoch=epoch
                        )
                        epoch += 1

    

