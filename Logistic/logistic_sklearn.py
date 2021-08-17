import numpy as np
from sklearn.linear_model import LogisticRegression

"""
函数说明:读取 txt 数据得到 特征矩阵和标签矩阵
Parameters:
	file_path   - 文件所在路径,可以是绝对路径也可以是相对路径
Returns:
	features    - 特征矩阵
	labels      - 标签矩阵
"""
def Load_data(file_path):
    fr = open(file_path)										#打开数据集
    features = []; labels = []
    for line in fr.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
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
	minVals = dataSet.min(0)    ## axis 指定成0，获得数据每个特征上的最小值
	maxVals = dataSet.max(0)    ## axis 指定成0，获得数据每个特征上的最大值
	ranges = maxVals - minVals  # 最大值和最小值的范围
	row = dataSet.shape[0]      # 返回dataSet的行数
	normDataSet = (dataSet - np.tile(minVals, (row, 1))) / np.tile(ranges, (row, 1))  # 原始值减去最小值除以最大和最小值的差,得到归一化数据
	return normDataSet, ranges, minVals


if __name__ == '__main__':
    trian_file_path = 'horseColicTraining.txt'
    trian_features, trian_labels = Load_data(trian_file_path)
    test_file_path = 'horseColicTest.txt'
    test_features, test_labels = Load_data(test_file_path)
    nom_trian_features = autoNorm(trian_features)[0]
    nom_test_features  = autoNorm(test_features)[0]

    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(nom_trian_features, trian_labels)
    test_accurcy = classifier.score(nom_test_features, test_labels ) * 100
    print('测试集正确率:%f%%' % test_accurcy)


