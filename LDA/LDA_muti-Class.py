import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


"""
虽然西瓜书放在了线性模型这一张，但是我觉得他放在降维哪儿更适合
输入：数据集 D = {(x1, y1),  (x2, y2), .... (xm, ym)}，其中任意样本 xi 为 n维向量， yi € {C1,  c2, ...Ck}，降维到的维度 d。
输出：降维后的样本集 D'

多类别：
1）计算每一种类别的样本均值，和整体的样本均值
2）计算类内散度矩阵Sw
3）计算类间散度矩阵 Sb   
4）计算矩阵Sw-1Sb
5) 计算 Sw-1Sb 的最大的 k个特征值和对应的 k个特征向量 （w1, w2, ... wk），得到投影矩阵 W
6）对样本集中的每一个样本特征 xi，转化为新的样本 zi = WTxi
7）得到输出样本集 D' = {(z1, y1),  (z2, y2), .... (zm, ym)}
"""

'''
函数说明:计算一类样本的均值点
Parameters:
    X - 数据集，样本矩阵
    y - label，标签矩阵
    k - 特征维数
Returns:
    topk_eig_vecs - 前k个特征值对应特征向量构成的矩阵，即时我们的w
'''
def LDA_dimensionality(X, y, k):
    label_ = list(set(y))    # set 去重，判断有多少个类别

    X_classify = {}          # 找出每个类别对应的所有样本，存成一个字典，键是类别，值是所有的样本
    for label in label_:
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == label])
        X_classify[label] = X1

    average = np.mean(X, axis=0)    # 整体样本矩阵的均值中心

    average_classify = {}           # 找出每个类别的均值中心，存成一个字典，同样键是类别，值是对应均值中心
    for label in label_:
        average1 = np.mean(X_classify[label], axis=0)
        average_classify[label] = average1

    #Sw = np.dot((X - average).T, X - average)
    Sw = np.zeros((len(average), len(average)))  # 计算类内散度矩阵，特征*特征个数
    for i in label_:
        Sw += np.dot((X_classify[i] - average_classify[i]).T, X_classify[i] - average_classify[i])

    # Sb=St-Sw
    Sb = np.zeros((len(average), len(average)))  # 计算类内散度矩阵
    for i in label_:
        Sb += len(X_classify[i]) * np.dot((average_classify[i] - average).reshape((len(average), 1)), (average_classify[i] - average).reshape((1, len(average))))

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))   # 计算Sw-1*Sb的特征值和特征矩阵

    sorted_indices = np.argsort(eig_vals)
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]  # 提取前k个特征向量

    return topk_eig_vecs



if '__main__' == __name__:

    iris = load_iris()
    X = iris.data
    y = iris.target

    W = LDA_dimensionality(X, y, 2)
    X_new = np.dot((X), W)
    plt.figure(1)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.title("自己按公式写的")

    # 与sklearn中的LDA函数对比
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    print(X_new)
    plt.figure(2)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.title("sklearn的")

    plt.show()