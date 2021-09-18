# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import svm

"""
数据结构，维护所有需要操作的值
Parameters：
    dataMatIn   - 数据矩阵
    classLabels - 数据标签
    C           - 松弛变量
    toler       - 容错率
"""
class optStruct:
	def __init__(self, dataMatIn, classLabels, C, toler):
		self.X = dataMatIn								#数据矩阵
		self.labelMat = classLabels						#数据标签
		self.C = C 										#松弛变量
		self.tol = toler 								#容错率
		self.m = np.shape(dataMatIn)[0] 				#数据矩阵行数，样本数
		self.alphas = np.mat(np.zeros((self.m,1))) 		#根据矩阵行数初始化alpha参数为0
		self.b = 0 										#初始化b参数为0
		self.eCache = np.mat(np.zeros((self.m,2))) 		#根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。


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
	Emax = calcEk(oS,max)
	maxDeltaE = 0

	oS.eCache[i] = [1,Ei]  									#根据Ei更新误差缓存
	validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]		#返回误差不为0的数据的索引值
	if (len(validEcacheList)) > 1:							#有不为0的误差
		for k in validEcacheList:   						#遍历,找到最大的Ek
			if k == i: continue 							#不计算i,浪费时间
			Ek = calcEk(oS, k)								#计算Ek
			deltaE = abs(Ei - Ek)							#计算|Ei-Ek|
			if (deltaE > maxDeltaE):						#找到maxDeltaE
				max = k; maxDeltaE = deltaE; Emax = Ek
	return max, Emax


"""
计算误差
Parameters：
    oS - 数据结构
    k  - 标号为k的数据
Returns:
    Ek - 标号为k的数据误差
"""
def calcEk(oS, k):
	fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
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
	Ek = calcEk(oS, k)										#计算Ek
	oS.eCache[k] = [1,Ek]									#更新误差缓存


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
优化的SMO算法
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
		j,Ej = selectJ(i, oS, Ei)         # 使用启发式选择alpha_j,并计算Ej
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
		eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
		if eta >= 0:
			print("eta>=0")
			return 0

		"""步骤4：更新alpha_j"""
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta

		"""步骤5：修剪alpha_j"""
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		updateEk(oS, j)       #更新Ej至误差缓存
		if (abs(oS.alphas[j] - alphaJold) < 0.00001):
			print("alpha_j变化太小")
			return 0

		"""步骤6：更新alpha_i"""
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		updateEk(oS, i)   # 更新Ei至误差缓存

		"""步骤7：更新b_1和b_2"""
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T

		"""步骤8：根据b_1和b_2更新b"""
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
			oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
			oS.b = b2
		else:
			oS.b = (b1 + b2)/2.0
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
Returns:
	oS.b        - SMO算法计算的b
	oS.alphas   - SMO算法计算的alphas
"""
def smoP(dataMatIn, classLabels, C, toler, maxIter):
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)					#初始化数据结构
	iter = 0 																						#初始化当前迭代次数
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)) :							#遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
		alphaPairsChanged = 0
		if entireSet:																				#遍历整个数据集
			for i in range(oS.m):
				alphaPairsChanged += innerL(i,oS)													#使用优化的SMO算法
				print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		else: 																						#遍历非边界值
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]						#遍历不在边界0和C的alpha
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		if entireSet:																				#遍历一次后改为非边界遍历
			entireSet = False                                                                       #若本轮是全集遍历，则下一轮进入边界遍历(下一轮while条件中的entire是False)
		elif (alphaPairsChanged == 0):																#如果alpha没有更新,计算全样本遍历
			entireSet = True                                                                        #若本轮是边界遍历，且本轮遍历未修改任何alpha对，则下一轮进入全集遍历
		print("迭代次数: %d" % iter)
	return oS.b,oS.alphas 																			#返回SMO算法计算的b和alphas


"""
函数说明:分类结果可视化,二维的情况
Parameters:
	dataMat  - 数据矩阵
	labelMat - 标签矩阵
    w        - 直线法向量或者说系数
    b        - 直线截距
Returns:
    无
"""
def showClassifer(dataMat,labelMat, w, b):
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
	b, alphas =  smoP(dataMat, labelMat, 10, 0.001, 40)
	w = np.dot((np.tile(labelMat.reshape(-1, 1), (1, 2)) * dataMat).T, alphas)    # 计算w
	showClassifer(dataMat, labelMat, w, b)

	x_test = np.array([3.6, 2.1])
	class_sample = classier(x_test, w, b)

"""我感觉完整的还没简单的好使，还巨不稳定，懵逼 ，不知道出了啥毛病，照着机器学习实战写的"""