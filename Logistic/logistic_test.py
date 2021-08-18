import numpy as np
import random

"""
函数说明:读取 txt 数据得到 特征矩阵和标签矩阵
Parameters:
	file_path   - 文件所在路径,可以是绝对路径也可以是相对路径
Returns:
	features    - 特征矩阵
	labels      - 标签矩阵
"""
def Load_data(file_path):
    fr = open(file_path)  # 打开数据集
    features = [];
    labels = []
    for line in fr.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        features.append(lineArr)
        labels.append(float(currLine[-1]))
    return np.array(features), np.array(labels)


"""
函数说明:对数据进行归一化
Parameters:
	dataSet     - 特征矩阵
Returns:
	normDataSet - 归一化后的特征矩阵
	ranges      - 数据范围
	minVals     - 数据最小值
"""
def autoNorm(dataSet):
    minVals = dataSet.min(0)  ## axis 指定成0，获得数据每个特征上的最小值
    maxVals = dataSet.max(0)  ## axis 指定成0，获得数据每个特征上的最大值
    ranges = maxVals - minVals  # 最大值和最小值的范围
    row = dataSet.shape[0]  # 返回dataSet的行数
    normDataSet = (dataSet - np.tile(minVals, (row, 1))) / np.tile(ranges, (row, 1))  # 原始值减去最小值除以最大和最小值的差,得到归一化数据
    return normDataSet, ranges, minVals


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
函数说明:将线性模型的输出隐射到s型函数,在通过阈值离散化
Parameters:
	lineOutput  - 线性模型的输出
Returns:
	classOutput - 分类结果输出
"""
def classifyVector(lineOutput):
    classOutput = sigmoid(lineOutput)  # 把负无穷到正无穷的映射到 0 到 1 之间
    classOutput[classOutput > 0.5] = 1
    classOutput[classOutput <= 0.5] = 0

    return classOutput


"""
函数说明:随机梯度算法

Parameters:
	dataMatrix  - 特征矩阵
	classLabels - 标签矩阵
	numIter     - 迭代次数
Returns:
	weights - 求得的回归系数数组(最优参数)
"""
def stocGradAscent(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    dataMatrix = np.concatenate((np.ones([m, 1]), dataMatrix), axis=1).astype(float)
    weights = np.ones(n + 1)  # 参数初始化#存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            del (dataIndex[randIndex])  # 删除已经使用的样本
    return weights


'''
函数说明:该函数用于梯度下降法计算线性回归的参数
Parameters:
	X             - 特征矩阵
	y             - 标签矩阵
	learning_rate - 学习率，也是最优化方法里面的步长，默认0.01
	epochs        - 迭代次数，默认为1000
Returns:
	w - 最佳的权重参数
'''
def grad_desc(X, y, learning_rate=0.01, epochs=1000):
    n = X.shape[0]  # 样本数量
    dim = X.shape[1] + 1  # 特征数量，+1是因为有常数项
    x = np.concatenate((np.ones([n, 1]), X), axis=1).astype(float)  # 同样由于有常数项，x矩阵需要多加一列1
    y = np.matrix(y).reshape(-1, 1).astype(float)  # y转化为行向量，方便矩阵运算
    w = np.zeros([dim, 1])  # 初始化参数

    for i in range(epochs):
        gradient = 2 * np.dot(x.transpose(), (sigmoid(np.dot(x, w)) - y)) / n
        w = w - learning_rate * gradient
    Acc, predict_y = predict(w, X, y)
    print('训练集分类精度', Acc)
    return w


'''
函数说明:该函数用于计算预测标签和精度
Parameters:
	w      - 最佳的权重参数
	test_X - 测试集特征矩阵
	test_y - 测试集标签矩阵
Returns:
	Acc       - 分类精度
	predict_y - 预测标签
'''
def predict(w, test_X, test_y):
    test_X = np.concatenate((np.ones([test_X.shape[0], 1]), test_X), axis=1).astype(float)
    test_y = np.array(test_y).reshape(-1, 1).astype(float)
    predict_y = classifyVector(np.dot(test_X, w))
    Acc = sum(np.array(predict_y).reshape(-1, 1) == test_y) / len(test_y);
    return Acc, predict_y


if __name__ == '__main__':
    trian_file_path = 'horseColicTraining.txt'
    trian_features, trian_labels = Load_data(trian_file_path)
    test_file_path = 'horseColicTest.txt'
    test_features, test_labels = Load_data(test_file_path)
    nom_trian_features = autoNorm(trian_features)[0]
    nom_test_features = autoNorm(test_features)[0]

    w1 = grad_desc(nom_trian_features, trian_labels, learning_rate=0.01, epochs=10000)
    Acc1, predict_y1 = predict(w1, nom_test_features, test_labels)
    print('测试集分类精度', Acc1)

    w2 = stocGradAscent(nom_trian_features, trian_labels, numIter=1000)
    Acc2, predict_y2 = predict(w2, nom_test_features, test_labels)
    print('测试集分类精度', Acc2)
"""
不用最小二乘发来求解参数而采用梯度法等优化算法的原因是因为样本过多最小二乘的计算量会成倍增加平方级增长，会很慢，而且还可能遇到特征矩阵不满秩的情况，求不出来
"""