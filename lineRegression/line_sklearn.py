import numpy as np
from sklearn import metrics
from sklearn import linear_model as line
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
	minVals = dataSet.min(0)    ## axis 指定成0，获得数据每个特征上的最小值
	maxVals = dataSet.max(0)    ## axis 指定成0，获得数据每个特征上的最大值
	ranges = maxVals - minVals  # 最大值和最小值的范围
	row = dataSet.shape[0]      # 返回dataSet的行数
	normDataSet = (dataSet - np.tile(minVals, (row, 1))) / np.tile(ranges, (row, 1))  # 原始值减去最小值除以最大和最小值的差,得到归一化数据
	return normDataSet, ranges, minVals


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
    y_train = y_train.reshape((-1, 1))               # 变成行向量
    y_test = y_test.reshape((-1, 1))


    model = line.LinearRegression()
    model.fit(X_train, y_train)
    pred_y = model.predict(X_test)
    print('使用sklearn线性模型函数的MSE：', metrics.mean_absolute_error(y_test, pred_y))
