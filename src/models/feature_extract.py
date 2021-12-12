import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import Counter

def kldb(p:float, q:float):
    """
    Kullback–Leibler divergence of binomial two distributions

    Kullback–Leibler divergence D(p || q) where p, q are parameters of two binomial distributions
    """
    return entropy([p, 1-p], qk=[q, 1-q])

def extract_words(N:int, alpha:float, y:pd.Series, body:pd.Series):
    """
    N: expect to extract N most informative words for NBC
    alpha: smoothing factor for laplace smoothing
    y: the labels
    body: corresponds to the contents in training set
    return the N words and corresponding priors in a list(maybe using yield)
    
    FORMAT: each element in the list is a tuple(word, negative prior, positive prior)
    PS: note that alpha impacts BOTH features selection and priors calculation
    """
    # C: "class", 
    # False: negative samples, 
    # True: positive samples, 
    # None: the whole population
    C = np.array([
        False, True, None
    ])   
    stat = dict()
    freq = dict()
    fq_sum = dict()
    kl = dict() # for Kullback–Leibler divergence

    for c in C:
        stat[c] = Counter()
    for item in body:
        stat[None].update(item.split(' '))
    for c in C[:2]:
        for item in body[y==c]:
            stat[c].update(item.split(' '))
        freq[c] = Counter({item: stat[c][item]+alpha for item in stat[None]})
        fq_sum[c] = sum(freq[c].values())
        freq[c] = Counter({item: freq[c][item]/fq_sum[c] for item in stat[None]})
        # OK, freq calculation finished    
    for item in stat[None]:
        # Kullback–Leibler divergence is not symmetric,
        # ham emails are reference points
        kl[item] = kldb(freq[False][item], freq[True][item])
    kl = pd.DataFrame(kl.items())
    kl.sort_values(by=1, inplace=True, ascending=False)

    for _, item in kl[:N].iterrows():
        yield(
            item.iloc[0],
            freq[False][item.iloc[0]],
            freq[True][item.iloc[0]]
        )
