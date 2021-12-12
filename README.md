# Naive Bayes Classifier
## 基本信息
姓名：赵晴岳
学号：
邮箱：zhaoqy19@mails.tsinghua.edu.cn

## 项目结构
- `src`：所有源文件，其中包括
  - `EDA`：探索性数据分析（一些初步统计尝试）其中`global_stat.ipynb`中包含了对邮件头、邮件体、数据集整体信息的统计
  - `utlis`：数据清洗和全局统计分析工具
  - `models`：特征提取&朴素贝叶斯
- `trec06p`：原始数据及经处理的数据，如欲复现实验结果，应将此文件夹放在`src`的同级目录
  - `data`：（**未打包上传的**）包含数据
  - `label`：（**未打包上传的**）包含标签
  - `inter`：（包含一些模型参数文件）前期统计数据&经全局预处理的数据
- `README.md`：本文件
- `Report.pdf`：实验报告

## 使用方法
> 如欲复现，则应在将`trec06p`文件夹放在`src`同级目录，在`trec06p`下建立`inter`文件夹，在`inter`文件夹下创建`records.csv`文件夹并添加字段（第一行）`LOWERCASE,N,PERCENTAGE,ALPHA,CRAFT,accuracy,precision,recall,f1_score`，并**提前**执行下列代码
> ```python
> import nltk
> nltk.download('wordnet')
> nltk.download('stopwords')
> nltk.download('punkt')
> ```
在上述准备完成后，在`src`目录下执行
```bash
python main.py
```
首先进行的是一次默认参数的实现（只有这次会输出朴素贝叶斯分类器五折交叉验证的变量列表），然后进行的是448次网格搜索，网格搜索所需时间较长！