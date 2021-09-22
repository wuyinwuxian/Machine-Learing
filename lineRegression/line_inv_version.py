import numpy as np
from sklearn import linear_model
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

class LinearRegression:
    def __init__(self):
        self.w = None             # 要训练的参数
        self.n_features = None    # 特征的个数

    def fit(self,X,y):            # 计算权重 W
        """
        w=(X^TX)^{-1}X^Ty
        """
        assert isinstance(X,np.ndarray) and isinstance(y,np.ndarray)
        assert X.ndim==2 and y.ndim==1    # assert 断言，就是说后面的条件成立时执行下面代码，不满足时返回错误
        assert y.shape[0]==X.shape[0]
        n_samples = X.shape[0]            #样本数量
        self.n_features = X.shape[1]      #特征个数，X.shape是个元组，元组的第二位代表列，也就是特征个数
        extra = np.ones((n_samples,))
        X = np.c_[X,extra]                #是在列方向扩展连接两个矩阵，就是把两矩阵左右相加，要求行数相等，相当于 hstack https://blog.csdn.net/qq_43657442/article/details/108030183
        if self.n_features < n_samples:
            self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # 其实就是X的广义逆乘Y，得到权重，详解见https://blog.csdn.net/qq_43657442/article/details/108032355
        else:
            raise ValueError('dont have enough samples')

    def predict(self, X):
        n_samples=X.shape[0]
        extra = np.ones((n_samples,))  #产生一个二维数组，用这样的方式代表这个数组可行向量可列向量
        X = np.c_[X, extra]
        if self.w is None:
            raise RuntimeError('cant predict before fit')
        y_=X.dot(self.w)
        return y_

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

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pre = lr.predict(X_test)

    mse = np.sqrt(np.average(np.square(y_pre - y_test)))



