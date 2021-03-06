# -*- coding: UTF-8 -*-
import numpy as np
from math import log
import collections
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

"""
函数说明:创建测试数据集
Parameters:
	无
Returns:
	dataSet - 数据集
	labels  - 特征标签
"""
def createDataSet():
    dataSet =[[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels                             # 返回数据集和特征标签

"""
函数说明:计算给定数据集的经验熵(香农熵)
Parameters:
	dataSet - 数据集
Returns:
	shannonEnt - 经验熵(香农熵)
"""
def calcShannonEnt(dataSet):
    number_sample = len(dataSet)                      # 返回数据集的行数
    labelCounts = {}                                  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:                           # 对每组特征向量进行统计
        currentLabel = featVec[-1]                    # 提取标签(Label)信息
        labelCounts[currentLabel] =labelCounts.get(currentLabel,0) +1        # Label计数
    shannonEnt = 0.0                                                         # 经验熵(香农熵)
    for key in labelCounts:                                                  # 计算香农熵
        prob = float(labelCounts[key]) / number_sample                       # 选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)                                    # 利用公式计算
    return shannonEnt                                                        # 返回经验熵(香农熵)



"""
函数说明: 提取在给定特征上取值 featureIndex 为 value 的样本组成新数据集
Parameters:
	dataSet         - 待划分的数据集
	featureIndex    - 用于提取数据集的特征
	value           - 提取特征取值等于 value 的样本
Returns:
    retDataSet      - 新的数据集   
"""
def splitDataSet(dataSet, featureIndex, value):
    retDataSet = []                                            # 创建返回的数据集列表
    for featVec in dataSet:                                    # 遍历数据集
        if featVec[featureIndex] == value:
            reducedFeatVec = featVec[:featureIndex]            # 去掉用来分割的特征
            reducedFeatVec.extend(featVec[featureIndex + 1:])
            retDataSet.append(reducedFeatVec)                  # 将符合条件的样本添加到返回的数据集
    return retDataSet                                          # 返回划分后的数据集


"""
函数说明:选择最优特征
Parameters:
	dataSet - 数据集
Returns:
	bestFeature - 信息增益最大的(最优)特征的索引值
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    Gain =[]                               # 存放每个特征的信息增益

    for feature in range(numFeatures):           # 遍历计算所有特征的信息增益
        featList = [example[feature] for example in dataSet]   # 获取 dataSet 的第i个特征的所有取值
        uniqueVals = set(featList)                             # 创建set集合{},元素不可重复，找到该特征有几种取值
        newEntropy = 0.0                                       # 经验条件熵

        for value in uniqueVals:                               # 计算第 i 个特征每一种取值上的熵
            subDataSet = splitDataSet(dataSet, feature, value) # 通过 特征的序号 i 和特征取值 value 提取出一个子数据集
            prob = len(subDataSet) / float(len(dataSet))       # 计算子集的占比
            newEntropy += prob * calcShannonEnt(subDataSet)    # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                    # 信息增益
        Gain.append(infoGain)

    bestFeatureIndex = Gain.index(max(Gain))
    return bestFeatureIndex                                    # 返回信息增益最大的特征的索引值


"""
函数说明:创建决策树

Parameters:
	dataSet    - 训练数据集
	labels     - 标签矩阵
	featLabels - 存储选择的最优特征标签
Returns:
	myTree     - 决策树
"""
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]                   # 取分类标签(是否放贷:yes or no)

    if classList.count(classList[0]) == len(classList):                # 如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:                       # 没有特征了
        return collections.Counter(classList).most_common(1)[0][0]     # 出现次数最多的标签即为最终类别, 可以参照官方文档 https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter.most_common

    bestFeat = chooseBestFeatureToSplit(dataSet)                       # 找到最优特征的索引
    bestFeatLabel = labels[bestFeat]                                   # 最优特征的标签，最有特征叫啥名
    featLabels.append(bestFeatLabel)

    myTree = {bestFeatLabel: {}}                                       # 根据最优特征的标签生成树
    del (labels[bestFeat])                                             # 删除已经使用特征标签，离散的就只用一次
    featValues = [example[bestFeat] for example in dataSet]            # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                       # 去掉重复的属性值，就得到这个节点要分支为几个

    for value in uniqueVals:                                           # 遍历特征，创建决策树。
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)   # 递归创建

    return myTree

"""
函数说明:使用决策树分类
Parameters:
	inputTree  - 已经生成的决策树
	featLabels - 存储选择的最优特征标签
	testVec    - 测试数据列表，顺序对应最优特征标签
Returns:
	classLabel - 分类结果
"""
def classify(inputTree, lensesLabels, testVec):
    firstStr = next(iter(inputTree))           # 获取当前结点，也就是用来分割数据的特征
    secondDict = inputTree[firstStr]           # 下一个字典
    featIndex = lensesLabels.index(firstStr)   # 特征所在列的索引
    classLabel = ''
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)    # 不断向下直到到达叶子节点
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    """测试构造的数据集，所谓的判断是否贷款给这个人"""
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)

    testVec = [0, 0,1,1]     # 测试数据
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')

    """测试隐形眼睛数据集，所谓的判断是否贷款给这个人"""
    lenses = []
    with open('lenses.txt', 'r') as fr:  # 加载文件
        for line in fr.readlines():
            temp = line.strip().split('\t')
            lenses.append(temp)          # 将样本加入数据集
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    featLabels = []
    myTree1 = createTree(lenses, lensesLabels.copy(), featLabels)     # 不用copy()的话 执行完创建树后会改变我的 lensesLabels，python这点做的不行，形参和实参分开多好

    test = ["young", "hyper", "yes","reduced"]     # 测试数据
    test_class = classify(myTree1, lensesLabels, test)

    """ 因为代码中划分数据是根据值来的划分的，所以只能做离散特征的问题，如果是连续的可以先区间离散化 """


