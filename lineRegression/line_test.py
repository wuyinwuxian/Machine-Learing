import numpy as np
from sklearn.datasets import load_diabetes



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
	minVals = dataSet.min(0)    # axis 指定成0，获得数据每个特征上的最小值
	maxVals = dataSet.max(0)    # axis 指定成0，获得数据每个特征上的最大值
	ranges = maxVals - minVals  # 最大值和最小值的范围
	row = dataSet.shape[0]      # 返回dataSet的行数
	normDataSet = (dataSet - np.tile(minVals, (row, 1))) / np.tile(ranges, (row, 1))  # 原始值减去最小值除以最大和最小值的差,得到归一化数据
	return normDataSet, ranges, minVals

'''
函数说明:该函数用于梯度下降法计算线性回归的参数
Parameters:
	X             - 特征矩阵
	y             - 标签矩阵
	learning_rate - 学习率，也是最优化方法里面的步长，默认0.01
	epochs        - 迭代次数，默认为1000
	eps           - Adagrad中使用的小正数防止了分母为零
	measure       - 采用那种方法来优化得到参数，默认传统梯度下降’gd’
Returns:
    w - 最佳的权重参数
'''
def grad_desc(X, y, learning_rate=0.01, epochs=1000, eps=0.0000000001, measure='gd'):
    n = X.shape[0]                                                    # 样本数量
    dim = X.shape[1] + 1                                              # 特征数量，+1是因为有常数项
    x = np.concatenate((np.ones([n, 1]), X), axis = 1).astype(float)  # 同样由于有常数项，x矩阵需要多加一列1
    y = np.matrix(y).reshape(-1, 1).astype(float)                     # y转化为行向量，方便矩阵运算
    w = np.zeros([dim, 1])                                            # 初始化参数

    if measure == 'gd':                                               # 常规的梯度下降法
        for i in range(epochs):
            loss = np.sum(np.power(np.dot(x, w) - y, 2))/n
            if (i % 1000 == 0):
                print(str(i) + ":" + str(loss))
            gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)/n
            w = w - learning_rate * gradient

    if measure == 'Adagrad':                                          # Adagrad法
        adagrad = np.zeros([dim, 1])
        for i in range(epochs):
            loss = np.sum(np.power(np.dot(x, w) - y, 2))/n
            if (i % 1000 == 0):
                print(str(i) + ":" + str(loss))
            gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)/n
            adagrad += np.square(gradient)
            w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    return w

'''
函数说明:该函数该函数用于计算预测标签和精度
Parameters:
	w      - 最佳的权重参数
	test_X - 测试集特征矩阵
	test_y - 测试集标签矩阵
Returns:
    mse       - 均方误差
    predict_y - 预测标签
'''
def predict(w, test_X, test_y):
    test_X = np.concatenate((np.ones([test_X.shape[0], 1]), test_X), axis=1).astype(float)
    test_y = np.matrix(test_y).reshape(-1, 1).astype(float)
    predict_y = np.dot(test_X, w)
    mse = np.sqrt(np.average(np.square(predict_y - test_y)))
    return mse, predict_y


if __name__ == '__main__':
    # 导入deabetes数据集，糖尿病数据集
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    nom_X, ranges, minVals = autoNorm(X)

    # 划分训练集和测试集
    offset = int(nom_X.shape[0] * 0.9)
    X_train, y_train = nom_X[:offset], y[:offset]
    X_test, y_test = nom_X[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))           # 变成行向量，方便运算，矩阵运算嘛
    y_test = y_test.reshape((-1, 1))

    w1 = grad_desc(X_train, y_train, learning_rate=0.03, epochs=10000, measure='gd')
    print('使用gd的MSE：', predict(w1, X_test, y_test)[0],'\n')

    w2 = grad_desc(X_train, y_train, learning_rate=0.03, epochs=10000, measure='Adagrad')
    print('使用Adagrad的MSE：', predict(w2, X_test, y_test)[0])
    """
    不用最小二乘发来求解参数而采用梯度法等优化算法的原因是因为样本过多最小二乘的计算量会成倍增加平方级增长，会很慢，而且还可能遇到特征矩阵不满秩的情况，求不出来
    """


