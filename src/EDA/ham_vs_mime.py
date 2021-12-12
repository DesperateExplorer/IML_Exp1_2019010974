"""
观察'MIME-Version: 1.0'的存在 与 ham or spam之间的关系
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../../trec06p/label/index', delimiter=' ', header=None)
mime = pd.read_csv('../../MIME-Version.csv')
t = [df[0], mime['mime-version: 1.0']]
jt = pd.DataFrame(data=t).transpose()

print(jt[0].value_counts()['spam']/jt[0].value_counts()['ham'])
print(jt[0][jt['mime-version: 1.0']==True].value_counts()['spam']/jt[0][jt['mime-version: 1.0']==True].value_counts()['ham'])
print(jt[0][jt['mime-version: 1.0']==False].value_counts()['spam']/jt[0][jt['mime-version: 1.0']==False].value_counts()['ham'])

sns.set_theme(style="whitegrid")
sns.catplot(data=jt, x=0, y='mime-version: 1.0',kind='bar',ci="sd", palette="dark", alpha=.4) 
plt.title("Percentage of \'MIME-Version: 1.0\' str", fontsize=16)
plt.show()