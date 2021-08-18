#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:加载数据，数据以[特征 ... 标签]方式存放，即最后一列是标签数据
Parameters:
    fileName - 文件名
Returns:
	dataMat  - 数据矩阵，以二维列表方式存放
"""
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')                  # 字符串分割成列表
		fltLine = list(map(float, curLine))					# 把字符串列表转化为float类型数值型列表
		dataMat.append(fltLine)
	return dataMat

"""
函数说明:绘制数据集，只限于二维，一个特征一个标签
Parameters:
	dataMat - 数据集合，形如[特征 ... 标签]
Returns:
	无
"""
def plotDataSet(dataMat):
	data = np.array(dataMat)
	fig = plt.figure()
	ax = fig.add_subplot(111)											#添加subplot
	ax.scatter(data[:,0], data[:,1], s = 20, c = 'blue',alpha = .5)		#绘制样本点, 数据是二维的，所以直接画就行
	plt.title('DataSet')												#绘制title
	plt.xlabel('X')
	plt.show()

"""
函数说明:根据特征切分数据集合
Parameters:
	dataSet - 数据集合
	feature - 切分的特征
	value   - 该特征的值
Returns:
	mat0 - 切分的数据集合0
	mat1 - 切分的数据集合1
"""
def binSplitDataSet(dataSet, feature, value):
	mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
	mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
	return mat0, mat1

"""
函数说明:生成叶结点
Parameters:
	dataSet - 数据集合
Returns:
	目标变量的均值作为叶节点的预测值
"""
def regLeaf(dataSet):
	return np.mean(dataSet[:,-1])

"""
函数说明:误差估计函数
Parameters:
	dataSet - 数据集合
Returns:
	目标变量的总方差
"""
def regErr(dataSet):
	return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

"""
函数说明:找到数据的最佳二元切分方式函数
Parameters:
	dataSet  - 数据集合
	leafType - 生成叶结点
	regErr   - 误差估计函数
	ops      - 用户定义的参数构成的元组
Returns:
	bestIndex - 最佳切分特征索引
	bestValue - 最佳特征值
"""
def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
	tolS = ops[0]; tolN = ops[1]                         # tolS允许的误差下降值,tolN切分的最少样本数

	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:       # 如果当前所有值相等,则最佳特征为none 最佳特征为均值（其实既然都一样随便取一个都行），这种情况在回归中基本不会出现
		return None, leafType(dataSet)

	m, n = np.shape(dataSet)                             # 统计数据集合的行m和列n
	S = errType(dataSet)                                 # 计算其误差估计
	bestS = float('inf'); bestIndex = 0; bestValue = 0   # 分别初始化最佳误差,最佳特征切分的索引值,最佳特征值

	for featIndex in range(n - 1):                       # 遍历所有特征
		for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):        # 遍历该特征的所有取值
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)    # 根据特征和特征值切分数据集
			if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # 如果分割后的数据少于tolN,则尝试下一个取值
				continue
			newS = errType(mat0) + errType(mat1)                          # 计算误差估计
			if newS < bestS:                                              # 如果误差估计更小,则更新特征索引值和特征值和误差
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS

	if (S - bestS) < tolS:                                                # 如果这轮分割的误差减少不大则不划分该数据集标记成叶子节点
		return None, leafType(dataSet)
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)           # 根据最佳的切分特征和特征值切分数据集合
	if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):          # 切分出的数据集很小，那也不划分该数据集标记成叶子节点
		return None, leafType(dataSet)

	return bestIndex, bestValue                                           # 返回最佳切分特征和特征值

"""
函数说明:树构建函数，树字典包含4个东西：分割的特征，该特征的取值，左子树，右子树
Parameters:
	dataSet  - 数据集合
	leafType - 建立叶结点的函数
	errType  - 误差计算函数
	ops      - 包含树构建所有其他参数的元组
Returns:
	retTree  - 构建的回归树
"""
def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)    # 选择最佳切分特征和特征值
	if feat == None:
		return val                                                  # 如果没有特征,则返回特征值，作为叶子节点的标签
	lSet, rSet = binSplitDataSet(dataSet, feat, val)                # 根据特征和特征取值分成左数据集和右数据集

	retTree = {}                                                    # 回归树
	retTree['分割特征'] = feat
	retTree['对应的分分割值'] = val
	retTree['left']  = createTree(lSet, leafType, errType, ops)     # 创建左子树和右子树
	retTree['right'] = createTree(rSet, leafType, errType, ops)

	return retTree

def isTree(obj):
	return (type(obj).__name__ == 'dict')

"""
函数说明:对树进行塌陷处理(即返回树平均值)
Parameters:
	tree - 树
Returns:
	树的平均值
"""
def getMean(tree):
	tree['right'] = getMean(tree['right'])
	tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right']) / 2.0

"""
函数说明:后剪枝
Parameters:
	tree - 训练好的树
	test - 测试集
Returns:
	tree - 剪枝后的树
"""
def prune(tree, testData):
	if np.shape(testData)[0] == 0:                                 # 如果测试集为空,则对树进行塌陷处理
		return getMean(tree)
	if (isTree(tree['right']) or isTree(tree['left'])):            # 如果有左子树或者右子树,则切分数据集
		lSet, rSet = binSplitDataSet(testData, tree['分割特征'], tree['对应的分分割值'])

	if isTree(tree['left']):                                       # 处理左子树(剪枝)
		tree['left'] = prune(tree['left'], lSet)
	if isTree(tree['right']):                                      # 处理右子树(剪枝)
		tree['right'] =  prune(tree['right'], rSet)

	if not isTree(tree['left']) and not isTree(tree['right']):       # 如果当前结点的左右结点为叶结点,那就判断可不可以合并
		lSet, rSet = binSplitDataSet(testData, tree['分割特征'], tree['对应的分分割值'])
		errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) + np.sum(np.power(rSet[:,-1] - tree['right'],2)) # 计算没有合并的误差
		treeMean = (tree['left'] + tree['right']) / 2.0              # 计算合并的均值
		errorMerge = np.sum(np.power(testData[:,-1] - treeMean, 2))  # 计算合并的误差
		if errorMerge < errorNoMerge:                                # 如果合并的误差小于没有合并的误差,则合并
			return treeMean
		else: return tree
	else: return tree

if __name__ == '__main__':
	print('剪枝前:')
	train_filename = 'ex2.txt'
	train_Data = loadDataSet(train_filename)
	train_Mat = np.mat(train_Data)

	tree = createTree(train_Mat)
	print(tree)
	print('\n剪枝后:')
	test_filename = 'ex2test.txt'
	test_Data = loadDataSet(test_filename)
	test_Mat = np.mat(test_Data)
	print(prune(tree, test_Mat))

	plotDataSet(train_Data)