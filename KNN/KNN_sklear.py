import numpy as np
import xlrd
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
函数说明:对数据进行归一化
Parameters:
	dataSet - 特征矩阵
Returns:
	normDataSet - 归一化后的特征矩阵
	ranges - 数据范围
	minVals - 数据最小值
"""
def autoNorm(dataSet):
	minVals = dataSet.min(0)    ## axis 指定成0，获得数据每个特征上的最小值
	maxVals = dataSet.max(0)    ## axis 指定成0，获得数据每个特征上的最大值
	ranges = maxVals - minVals  # 最大值和最小值的范围
	row = dataSet.shape[0]      # 返回dataSet的行数
	normDataSet = (dataSet - np.tile(minVals, (row, 1))) / np.tile(ranges, (row, 1))  # 原始值减去最小值除以最大和最小值的差,得到归一化数据
	return normDataSet, ranges, minVals

"""
函数说明:读取数据函数，算是算法之外的部分
Parameters:
	path - 数据所在目录（在当前目录下，可直接使用文件名，不然则应该为绝对路径）
Returns:
	datamatrix - 数据形式的数据
"""
def excel2matrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for i in range(ncols):
        cols = table.col_values(i)
        datamatrix[:, i] = cols
    return datamatrix


if __name__ == "__main__":
    data = excel2matrix("data.xls")
    lables = excel2matrix("labels.xls").astype(int)  # 标签是整数比较好看。清爽

    nom_data, ranges, minVals = autoNorm(data)
    class_number = 3

    test = np.array([0.1,0.4,0.3]).reshape(1,-1)    # 你随便构造一个新样本喽，每个属性都归一化了，所以你的属性取值应该在0-1之间

    neigh = kNN(n_neighbors=3, algorithm='auto')    # 构建kNN分类器
    neigh.fit(nom_data, lables.flatten())           # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    test_class = neigh.predict(test)                # 预测输出
    """
    使用这种方法，X没说的（样本数，特征数），但对标签Y有要求，要把Y展开成一个一维向量（这儿可能是我理解差了点，
    似乎在python里面一维的既可以是列向量也可以是行向量），我在导入数据的时候是把Y 变成（样本数，1）的二维数组
    """