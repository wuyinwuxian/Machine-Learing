import numpy as np
import xlrd
import collections

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
函数说明:计算向量与向量（矩阵多个向量）的距离
Parameters:
	new_data - 向量
	base_data - 矩阵或者向量
Returns:
	distances - 距离（单个值或者向量形式）
"""
def calculate_distances(new_data,base_data):
	dataSize  = base_data.shape[0]                              # numpy函数shape[0]返回dataSet的行数
	diffMat   = np.tile(new_data, (dataSize, 1)) - base_data    # 在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
	sqDiffMat = diffMat ** 2                                    # 每个元素后平方
	distances = (sqDiffMat.sum(axis=1)) ** 0.5                  # sum()所有元素相加,sum(0)列相加,sum(1)行相加 开方,计算出距离
	return distances

"""
函数说明:kNN算法,分类器
Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果
"""
def classify0(inX, dataSet, labels, k):
	distances = calculate_distances(inX,dataSet)                          # 计算距离
	k_labels = [labels[index][0] for index in distances.argsort()[0: k]]     # 距离最近的前k个样本的标签
	label = collections.Counter(k_labels).most_common(1)[0][0]            # 出现次数最多的标签即为最终类别
	return label

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

	test = np.array([0.1,0.4,0.3]).reshape(1, -1)   # 你随便构造一个新样本喽，每个属性都归一化了，所以你的属性取值应该在0-1之间
	test_class = classify0(test, nom_data, lables, class_number)