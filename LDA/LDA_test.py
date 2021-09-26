import numpy as np

'''
函数说明:产生几个类别的随机样本
Parameters:
	无
Returns:
    Xi   -  第i类样本的组成的矩阵
'''
def createDataSet():
    # 类别1
    X1 = np.random.random((80, 4)) * 5 + 15
    # 类别2
    X2 = np.random.random((60, 4)) * 5 + 2
    return X1, X2

'''
函数说明:计算一类样本的均值点
Parameters:
	dataset - 某个类别的样本矩阵
Returns:
    np.array(ave) - 这一类样本的中心（各个维度上的均值）
'''
def average(dataset):
    ave = np.mean(dataset, axis=0)
    return np.array(ave)

'''
函数说明:计算一类样本类内散度
Parameters:
	dataset - 某个类别的样本矩阵
	ave     - 这一类样本的中心（各个维度上的均值）
Returns:
    sw      - 类内散度
'''
def compute_sw(dataset, ave):
    n = dataset.shape[1]
    sw = np.zeros((n, n))
    for line in dataset:
        x = np.array(line - ave).reshape(-1,1)
        sw += np.dot(x,x.T)
    return sw



if __name__ == '__main__':
    """生成数据"""
    X1, X2 = createDataSet()

    """计算每一种类别的样本均值"""
    u1 = average(X1)
    u2 = average(X2)

    """计算类内散度矩阵（每个都是特征*特征大小）"""
    x1_sw = compute_sw(X1, u1)
    x2_sw = compute_sw(X2, u2)
    Sw = x1_sw + x2_sw

    """求广义逆,再和均值之差相乘"""
    pinv = np.linalg.pinv(Sw)
    w = np.dot(pinv, u1 - u2)


"""
虽然西瓜书放在了线性模型这一张，但是我觉得他放在降维哪儿更适合
输入：数据集 D = {(x1, y1),  (x2, y2), .... (xm, ym)}，其中任意样本 xi 为 n维向量， yi € {C1,  c2, ...Ck}，降维到的维度 k。
输出：降维后的样本集 D'

二类别：
1）计算每一种类别的样本均值
2）计算类内散度矩阵Sw
3) 计算类Sw-1(u1-u2) = w 得到投影的线的权重（斜率） W
4）对样本集中的每一个样本特征 xi，转化为新的样本 zi = WTxi
5）得到输出样本集 D' = {(z1, y1),  (z2, y2), .... (zm, ym)}
"""
