# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random

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
函数说明:数据可视化
Parameters:
    dataMat  - 数据矩阵
    labelMat - 数据标签
Returns:
    无
"""
def showDataSet(dataMat, labelMat):
    data_p = dataMat[labelMat==1]     # positive
    data_n = dataMat[labelMat==-1]    # negative
    plt.scatter(np.transpose(data_p)[0], np.transpose(data_p)[1])  # 正样本散点图
    plt.scatter(np.transpose(data_n)[0], np.transpose(data_n)[1])  # 负样本散点图
    plt.show()


"""
函数说明:随机选择alpha
Parameters:
    i - alpha_i的索引值
    m - alpha参数个数
Returns:
    j - alpha_j的索引值
"""
def selectJrand(i, m):
    j = i  # 选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j


"""
函数说明:对 alpha 进行上下界约束
Parameters:
    aj - alpha_j值
    H  - alpha上限
    L  - alpha下限
Returns:
    aj - alpah_j值
"""
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

"""
函数说明:简化版SMO算法,来计算b 和
Parameters:
    dataMatIn   - 数据矩阵
    classLabels - 数据标签
    C           - 松弛变量
    toler       - 容错率
    maxIter     - 最大迭代次数
Returns:
    b      - 直线截距
    alphas - 拉格朗日系数矩阵
"""
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 转换为numpy的mat存储
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()

    m, n = np.shape(dataMatrix)
    b = 0                                           # 初始化b参数
    alphas = np.mat(np.zeros((m, 1)))               # 初始化alpha参数，设为0
    iter_num = 0                                    # 初始化迭代次数

    while (iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            '''步骤1：计算误差Ei  (fxi - yi) fxi = wx+b 转换成 fxi = alphas*yi*xi.T*x +b'''
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])

            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):  # alpha_i是不是已经很好了
                j = selectJrand(i, m)                   # 随机选择另一个与alpha_i成对优化的alpha_j

                """步骤1：计算误差Ej"""
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()            # 保存更新前的aplpha值，使用深拷贝
                alphaJold = alphas[j].copy()

                """步骤2：计算上下界L和H"""
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue

                """步骤3：计算eta"""
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j,:] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                """步骤4：更新alpha_j"""
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta

                """步骤5：修剪alpha_j"""
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("alpha_j变化太小,alpha_j = ", float(alphas[j]))
                    continue

                """步骤6：更新alpha_i"""
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                """步骤7：更新b_1和b_2"""
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T

                """步骤8：根据b_1和b_2更新b"""
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1          # 统计优化次数
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))  # 打印统计信息
        # 更新迭代次数
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b, alphas


"""
函数说明:分类结果可视化,二维的情况
Parameters:
	dataMat - 数据矩阵
    w       - 直线法向量或者说系数
    b       - 直线截距
Returns:
    无
"""
def showClassifer(dataMat, labelMat, w, b):
    data_p = dataMat[labelMat == 1]   # positive
    data_n = dataMat[labelMat == -1]  # negative
    plt.scatter(np.transpose(data_p)[0], np.transpose(data_p)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_n)[0], np.transpose(data_n)[1], s=30, alpha=0.7)  # 负样本散点图

    """绘制直线"""
    a1, a2 = w[0, 0], w[1, 0]
    b = b[0,0]                                             # 返回的的 b 是一个数据的数组
    x1, x2 = np.max(dataMat[:,0]), np.min(dataMat[:,0])    # 因为是二维的，所以取第一维的最大最小值，分别计算对应的第二维的取值构成两个点来画直线
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])

    """找出支持向量点"""
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

"""
函数说明:对单个新样本进行分类，多个情况自行封装即可
Parameters:
	x - 待分类数据
    w - 直线法向量或者说系数
    b - 直线截距
Returns:
    class_sample - 类别
"""
def classier(x,w,b):
    res = np.dot(x,w) + b
    class_sample = 1 if res[0,0]>0 else -1
    return class_sample

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    # showDataSet(dataMat, labelMat)
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = np.dot((np.tile(labelMat.reshape(-1, 1), (1, 2)) * dataMat).T, alphas)    # 计算w
    showClassifer(dataMat,labelMat, w, b)

    x_test = np.array([3.6,2.1])
    class_sample = classier(x_test, w, b)




