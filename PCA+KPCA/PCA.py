import pandas as pd
import numpy as np

df = pd.read_csv('wine.data', header=None)
X, y = df.values[:, 1:], df.values[:, 0]

# step 1  去中心化
X -= X.mean(0)

## step 2   算协方差矩阵
N = X.shape[0]
C = X.T.dot(X)/N

## step 3  特征值分解
Lambda, Q = np.linalg.eig(C)  # Lambda 是特征值， Q 的每一列是其对应的特征向量，为啥是列，别人就是这么规定的  https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html

## step 4  选前k个
k = 3
index =  np.argsort(Lambda)[::-1]    # 先排序后反转，变成从大到小，不能指定reverse就很难受
index_k = index[0:k]                 # 把前k个的索引找到
W = Q[:,index_k]                     # 这就是降维矩阵

## step 5
X_pca = X.dot(W)