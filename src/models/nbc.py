import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from .feature_extract import extract_words

class NBC:
    """
    Naive Bayes Classifier for binary classification, 
    where positive samples are labelled as **True**, and negative samples are labelled as **False**.
    """
    def __init__(self, alpha, percentage, hand_crafted:bool, N=15) -> None:
        """
        alpha: the smoothing factor of laplace smoothing
        percentage: the percentage of training set to use
        hand_crafted: whether hand-crafted features are used

        N is the expected number of discriminative words
        4 is the number of hand-crafted features, which is fixed in this project

        PS1: mesh_index is between 0 and 1, and higher means the metadata is rougher

        PS2: The default value of N=15 is from [Better Bayesian Filtering](http://www.paulgraham.com/better.html)
        """
        self.N = N
        self.feat_cnt = 4 # see the comments above
        self.alpha = alpha
        self.percent = percentage
        self.craft = hand_crafted

    def fit(self, yX: pd.DataFrame):
        """
        gain a tuple (list of log priors, dict of discriminative words, discriminative value of mess)
        """
        yX.reset_index(drop=True, inplace=True)
        yX = yX.loc[:int(len(yX)*self.percent), :].copy()
        self.y = yX.loc[:, 'label']
        self.quant = np.mean(yX.mess_index) # quant means quantile
        yX.drop('label', axis=1, inplace=True)
        self.samples = (yX.loc[self.y==False, :], yX.loc[self.y==True, :])
        self.priors = defaultdict(list)

        for c in [False, True]:
            for field in yX.columns[:self.feat_cnt-1]:
                self.priors[c].append(
                    np.sum(self.samples[c][field])/len(self.samples[c][field])
                )        
            # deal with mess_index
            self.priors[c].append(
                np.exp(np.sum(self.samples[c]['mess_index'] > self.quant)/len(self.samples[c]['mess_index']))
            )

        # find discriminative words
        self.words = [] # list of discriminative words
        for word, F, T in extract_words(self.N,self.alpha, self.y, yX.loc[:, 'body']):
            self.priors[False].append(F)
            self.priors[True].append(T)
            self.words.append(word)
        
        self.log_priors = {item: np.log(self.priors[item]) for item in self.priors}

    def predict(self, X:pd.DataFrame):
        """
        X: the feature matrix of test set WITH LABELS REMOVED
        return an array of preditions
        """
        X.reset_index(inplace=True, drop=True)
        pred = dict()
        pred[False] = np.zeros(len(X))
        pred[True] = np.zeros(len(X))
        temp = Counter()

        for c in [False, True]:
            for ind, row in X.iterrows():   
                # utf-8, html, consistent             
                pred[c][ind] += self.craft * np.sum(
                    [row.iloc[k]*self.log_priors[c][k] + (1-row.iloc[k])*np.log(1 - self.priors[c][k]) for k in range(self.feat_cnt-1)]
                    )
                # mess_index
                pred[c][ind] += self.craft * (
                    (row.iloc[self.feat_cnt-1] > self.quant)*np.log(self.log_priors[c][self.feat_cnt-1]) + (row.iloc[self.feat_cnt-1] <= self.quant)*np.log(1 - self.log_priors[c][self.feat_cnt-1])
                    )
                # informative words
                temp.clear()
                temp.update(row.iloc[self.feat_cnt].split(' '))

                # utilize the feature of collections.Counter():
                # if word is not in temp, then temp[word] is 0
                cnt = np.array([temp[word] for word in self.words])
                pred[c][ind] += np.sum(cnt*self.log_priors[c][self.feat_cnt:])
        
        return pred[True] > pred[False]
    
    def info(self) -> dict:
        """
        return some basic hyperparameters
        """
        return {
            'N': self.N,
            'PERCENTAGE': self.percent,
            'ALPHA': self.alpha,
            'CRAFT': self.craft
        }
    def informative_words(self):
        return self.words