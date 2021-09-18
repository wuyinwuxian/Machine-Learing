# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import svm

"""
函数描述： 核函数值的计算，只写了 线性核函数 和 RBF 输入其他的就返回错误信息
Parameters：
    X     - 整个样本矩阵  （样本数量*特征个数）
    Xi    - 第i个样本构成的向量  （1*特征个数）
    kTup  - 核函数信息，是一个元组，里面包含采用的核的类型以及核函数需要的参数
返回值：
    K     - 样本 Xi 和其他所有样本的核函数值
"""
def kernelTrans(X, Xi, kTup):  # calc the kernel or transform data to a higher dimensional space
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * Xi.T   # linear kernel 线性核就直接做内积
    elif kTup[0] == 'rbf':              # RBF 核 就计算每个样本与 Xi 的欧氏距离（对应维度相减再平方）  然后利用径向基函数算一遍
        for j in range(m):
            deltaRow = X[j, :] - Xi
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1]**2))  # 就得到了  Xi 和其他所有样本的核函数值
    else:
        raise NameError('该核函数没有实现，请选择（lin or rbf）')
    return K


"""
数据结构，维护所有需要操作的值
Parameters：
    dataMatIn   - 数据矩阵
    classLabels - 数据标签
    C           - 松弛变量
    toler       - 容错率
    kTup        - 核函数信息，是一个元组，里面包含采用的核的类型以及核函数需要的参数
"""
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn								# 数据矩阵
        self.labelMat = classLabels						# 数据标签
        self.C = C 										# 松弛变量
        self.tol = toler 								# 容错率
        self.m = np.shape(dataMatIn)[0] 				# 数据矩阵行数，样本数
        self.alphas = np.mat(np.zeros((self.m, 1))) 		# 根据矩阵行数初始化alpha参数为0
        self.b = 0 										# 初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m, 2))) 		# 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        """注意，以下三行是核SVM中添加的，注意核之前的完整版的做个对比"""
        self.K = np.mat(np.zeros((self.m, self.m)))     # 核函数据矩阵，（样本*样本）
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)    # 初始化的时候计算好所有的核函数值（其实就是两个样本映射到高维后的内积）


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
    data_p = dataMat[labelMat == 1]     # positive
    data_n = dataMat[labelMat == -1]    # negative
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
启发式选取另一个alpha 启发式是 max|Ei-Ek| 找到这个k，这个也是根据更新公式来的
Parameters：
    i  - 标号为i的数据的索引值
    oS - 数据结构
    Ei - 标号为i的数据误差
Returns:
    max    - 误差最大的那个数据的索引值
    Emax   - 最大的数据误差
or
    rand   - 随机数据的索引值
    Erand  - 随机数据对应的的数据误差
"""
def selectJ(i, oS, Ei):
    max = selectJrand(i, oS.m)                              # 先随机选一个如果没有就返回随机的
    Emax = calcEk(oS, max)
    maxDeltaE = 0

    oS.eCache[i] = [1, Ei]  # 根据Ei更新误差缓存
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:  # 遍历,找到最大的Ek
            if k == i:
                continue  # 不计算i,浪费时间
            Ek = calcEk(oS, k)  # 计算Ek
            deltaE = abs(Ei - Ek)  # 计算|Ei-Ek|
            if (deltaE > maxDeltaE):  # 找到maxDeltaE
                max = k
                maxDeltaE = deltaE
                Emax = Ek
    return max, Emax


"""
计算误差, 同样的由于是里面用到了样本内积，所以需要做点更改
Parameters：
    oS - 数据结构
    k  - 标号为k的数据
Returns:
    Ek - 标号为k的数据误差
"""
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


"""
计算Ek,并更新误差缓存
Parameters：
    oS - 数据结构
    k  - 标号为k的数据的索引值
Returns:
    无
"""
def updateEk(oS, k):
    Ek = calcEk(oS, k)  # 计算Ek
    oS.eCache[k] = [1, Ek]  # 更新误差缓存


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
优化的SMO算法，因为我们这是核函数的版本。所以需要一点点修改
Parameters：
	i - 标号为i的数据的索引值
	oS - 数据结构
Returns:
	1 - 有任意一对alpha值发生变化
	0 - 没有任意一对alpha值发生变化或变化太小
"""
def innerL(i, oS):
    """步骤1：计算误差Ei"""
    Ei = calcEk(oS, i)

    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):   # alpha_i是不是已经很好了
        j, Ej = selectJ(i, oS, Ei)         # 使用启发式选择alpha_j,并计算Ej
        alphaIold = oS.alphas[i].copy()   # 保存更新前的aplpha值，使用深拷贝
        alphaJold = oS.alphas[j].copy()

        """步骤2：计算上下界L和H"""
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0

        """步骤3：计算eta"""
        """注意：我们计算 eta 是用到了两个样本的内积，我们用来核函数，那么映射后的两个样本的内积我们已经计算过了，所以要修改"""
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0

        """步骤4：更新alpha_j"""
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta

        """步骤5：修剪alpha_j"""
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # 更新Ej至误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0

        """步骤6：更新alpha_i"""
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)   # 更新Ei至误差缓存

        """步骤7：更新b_1和b_2"""
        """注意：同样在计算 b 的时候我们用到了样本的内积，所以也要更换成计算好了 K """
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

        """步骤8：根据b_1和b_2更新b"""
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


"""
完整的线性SMO算法
Parameters：
	dataMatIn   - 数据矩阵
	classLabels - 数据标签
	C           - 松弛变量
	toler       - 容错率
	maxIter     - 最大迭代次数
	kTup        - 核函数信息，是一个元组，里面包含采用的核的类型以及核函数需要的参数
Returns:
	oS.b        - SMO算法计算的b
	oS.alphas   - SMO算法计算的alphas
"""
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler,kTup)  # 初始化数据结构
    iter = 0  # 初始化当前迭代次数
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entireSet:  # 遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 使用优化的SMO算法
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # 遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:  # 遍历一次后改为非边界遍历
            entireSet = False  # 若本轮是全集遍历，则下一轮进入边界遍历(下一轮while条件中的entire是False)
        elif (alphaPairsChanged == 0):  # 如果alpha没有更新,计算全样本遍历
            entireSet = True  # 若本轮是边界遍历，且本轮遍历未修改任何alpha对，则下一轮进入全集遍历
        print("迭代次数: %d" % iter)
    return oS.b, oS.alphas  # 返回SMO算法计算的b和alphas


"""
函数说明:测试核函数版本的SVM
Parameters:
	kTup - 核函数信息，是一个元组，里面包含采用的核的类型以及核函数需要的参数
Returns:
    无
"""
def testRbf(kTup):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 20, 0.0001, 10000, kTup)  #  利用SMO 把b 和 alphas 计算出来
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()

    """找支持向量"""
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]  # get matrix of only support vectors
    labelSV = labelMat[svInd]

    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)

    errorCount = 0
    for i in range(m):
        """这两行就是对一个新样本的预测过程"""
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)   # 计算他们和支持向量的核函数值
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        """
        上面一行代码就是带入公式, SVM 的一个优点不就是只用支持向量就可以完成预测吗，
        就是那个 支持向量和待预测样本内积然后乘对应支持向量标签和alpha 求和，再加b 的公式
        """
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    showClassifer(dataArr, labelArr, alphas)


    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


"""
函数说明:看看支持向量的情况
Parameters:
	dataMat  - 数据矩阵
	labelMat - 标签矩阵
    alphas   - 支持向量标记矩阵
Returns:
    无
"""
def showClassifer(dataMat,labelMat,alphas):
    data_p = dataMat[labelMat == 1]   # positive
    data_n = dataMat[labelMat == -1]  # negative
    plt.scatter(np.transpose(data_p)[0], np.transpose(data_p)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_n)[0], np.transpose(data_n)[1], s=30, alpha=0.7)  # 负样本散点图

    """找出支持向量点"""
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


if __name__ == '__main__':
    kernal =  ('rbf', 1.2)   # 后面的1.2是rbf 函数里面的 sigma， 分母上那玩意儿
    testRbf(kernal)

"""
相比较完整版的，核 SVM 就改动了几个地方
1、多定义一个核转换矩阵，用于计算 一个样本核目标样本矩阵（训练是所有的样本矩阵，测试是支持向量构成的样本矩阵）的核函数值
2、优化操作里面多了一 K （核函数值矩阵，样本*样本大小），然后初始化的时候计算一遍，把 K 计算好
3、在 SMO 具体过程 innerL 函数里面，把涉及到计算内积的都改成直接从 K 里面取即可，毕竟我们算过了嘛
4、同样，在计算误差的时候，也有内积，所以也换成从 K 里面取
5、函数的参数，会多一个 kTup ，是一个元组，里面包含采用的核的类型以及核函数需要的参数
6、在预测的时候，之前我们是有 w 和 b ，非线性的情况下是算不出来 w 的，所以我们用了支持向量预测的那个公式，
   其实在先行情况下，你用支持向量那个公式也是可以的，因为那就是把 w 用 alpha y X表示替换了而已。没区别的，
   w 的表示是对朗格朗日函数求导就行了 （周志华《机器学习》公式6.9）
应该差不多就这些，其他的可能有细微改动，我不太记得了，相较于完整版的线性的情况，整个非线性班的还可以
"""

