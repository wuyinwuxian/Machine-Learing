# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import svm

"""
函数说明:读取数据
Parameters:
    fileName - 文件名
Returns:
    dataMat  - 数据矩阵
    labelMat - 数据标签
"""
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    return np.array(dataMat), np.array(labelMat)

"""
函数说明:分类结果可视化,二维的情况
Parameters:
	dataMat - 数据矩阵
    w       - 直线法向量或者说系数
    b       - 直线截距
Returns:
    无
"""
def showClassifer(dataMat,labelMat,support_vector):
    data_p = dataMat[labelMat == 1]   # positive
    data_n = dataMat[labelMat == -1]  # negative
    plt.scatter(np.transpose(data_p)[0], np.transpose(data_p)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_n)[0], np.transpose(data_n)[1], s=30, alpha=0.7)  # 负样本散点图

    """找出支持向量点"""
    for vector in support_vector:
        x, y = vector[0],vector[1]
        plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()



if __name__ == '__main__':
	dataMat, labelMat = loadDataSet('testSetRBF.txt')
	classifier = svm.SVC(C=20, kernel='rbf', gamma=2.5, decision_function_shape='ovo')  # ovo:一对一策略，线性核才有 w 和 b，方便画图
	classifier.fit(dataMat, labelMat)
	showClassifer(dataMat, labelMat, classifier.support_vectors_)


	x_test = np.array([[8.6, 2.1]])
	class_sample =classifier.predict(x_test)
