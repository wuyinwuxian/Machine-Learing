import numpy as np


"""
函数说明:计算类别概率矩阵，其实就是计算每一类占总样本的多少。类先验概率，分子上的那个P(c)
Parameters:
    y - 标签矩阵
Returns:
    P - 类别概率矩阵
"""
def compute_class_probability(y):
    Y_ = list(set(y))
    sample_class_number = len(Y_)    # 这儿多此一举是预防类别编码不是从0开始的，而是其他的
    P = np.zeros((sample_class_number,1))
    for i in range(sample_class_number):
        P[i] = (len(y[y ==Y_[i]]) + 1) / (len(y) + sample_class_number)

    return P

"""
函数说明:计算概率字典，键是属性，值是这个属性各个取值上的类条件概率，每个属性对应一个二维数据feature_P（feature_P[i][j] 代表再当前属性上的第j个取值再第i个类别上的条件概率，
        用人话来说就是第i个类别的样本中这个属性取值是j的样本有多大的比例），这是离散的情况，连续的情况就计算一个均值和方差出来，确定好概率密度函数
Parameters:
    X                   - 样本数据矩阵
    y                   - 标签矩阵
    is_discrete_feature - 是否是离散属性 ，特征*1 大小
Returns:
    feature_probability_dict - 条件概率字典
"""
def compute_feature_probability_dict(X,y,is_discrete_feature):
    feature_probability_dict = {}
    class_number = len(set(y))   # 类别数

    for feature_index in range(X.shape[1]):                                   # 对于每一个属性
        if is_discrete_feature[feature_index] == True:
            feature_probability_dict[feature_index] = compute_discrete(X, y, feature_index, class_number)
        else:
            feature_probability_dict[feature_index] = compute_continue(X, y, feature_index, class_number)
    return  feature_probability_dict

"""
函数说明: 计算离散属性的 条件概率矩阵，feature_P[i][j] 代表再当前属性上的第j个取值再第i个类别上的条件概率，用人话来说就是第i个类别的样本中这个属性（feature）上取值是j的样本有多大的比例
Parameters:
    X              - 样本数据矩阵
    y              - 标签矩阵
    feature_index  - 当前特征索引
    class_number   - 类别数
Returns:
    feature_P    - 特征条件概率矩阵 （类别数*特征可能取值个数）
"""
def compute_discrete(X,y,feature_index,class_number):
    feature_values = np.unique(X[:, feature_index])  # 当前特征可能有多少个取值
    feature_P = np.zeros((class_number, len(feature_values)))
    feature_values_number = len(feature_values)

    for class_index in range(class_number):
        D_c = np.where(y == class_index)                                                # 类别是 sample_class 的样本的下标（索引）
        for feture_value_index in range(feature_values_number):
            D_xi = np.where(X[:, feature_index] == feature_values[feture_value_index])  # 当前属性上取值是 feature_value 的样本的下标（索引）
            D_cxi = len(np.intersect1d(D_xi,D_c))                                       # 取两个索引的交集，那就是  sample_class 类别的样本中。当前属性取值为 feature_value 的样本的下标索引，然后取len ，那就是个数
            feature_P[class_index][feture_value_index] = (D_cxi + 1) / (len(D_c[0]) + feature_values_number)
    return feature_P


"""
函数说明: 计算连续属性的 条件概率矩阵，就计算出来一个均值和方差
Parameters:
    X              - 样本数据矩阵
    y              - 标签矩阵
    feature_index  - 当前特征索引
    class_number   - 类别数
Returns:
    feature_P_fuction      - 特征条件密度函数（[均值,方差]）矩阵
"""
def compute_continue(X,y,feature_index,class_number):
    feature_P_fuction = np.zeros((class_number, 2))   # 只需要均值和方差，两列就可以
    for class_index in range(class_number):
        X_c = X[np.where(y == class_index),feature_index]
        mu = np.mean(X_c)     # 均值  mu(拼音)
        sigma = np.std(X_c)   # 方差  sigma (拼音)
        """上面两句写出来只是方便代码阅读，你也可以精简直接计算赋值到矩阵里面"""
        feature_P_fuction[class_index][0] = mu
        feature_P_fuction[class_index][1] = sigma
    return feature_P_fuction


"""
函数说明: 预测一个样本的类别
Parameters:
    x                         - 单个样本数据
    P                         - 类别概率矩阵
    feature_probability_dict  - 条件概率字典
    is_discrete_feature       - 是否是离散属性 ，特征*1 大小
Returns:
    predict_class - 预测的类别
"""
def predict(x,P,feature_probability_dict,is_discrete_feature):
    class_number = P.shape[0]
    result_P = np.ones_like(P)
    for class_index in range(class_number):        # 对每一个类别都计算一个可能的预测概率
        result_P[class_index] *= P[class_index]
        for j in range(len(is_discrete_feature)):  # 对每一个类别拿到类条件概率
            if is_discrete_feature[j]==True:       # 离散属性就从计算好了的字典力去找
                """ 注意：这儿 int(x[j]) 可以充当索引下标的前提是，离散值我们用0开始的自然数进行编码了，如果不是这样的，请先对数据进行编码，或者从新设计一套直接处理字符串的算法也可以"""
                feture_value_index = int(x[j])
                Conditional_Probability = feature_probability_dict[j][class_index][feture_value_index]
            else:                               # 连续属性就把计算好的 sigma 和 mu 拿出来，带入正态分布的概率密度函数算概率
                mu = feature_probability_dict[j][class_index][0]
                sigma = feature_probability_dict[j][class_index][1]
                Conditional_Probability = (1./(np.sqrt(2*np.pi)*sigma)) * (np.exp(-(x[j]-mu)**2/(2*sigma**2)))
            result_P[class_index] *= Conditional_Probability

    """ 返回最大的概率下标就是类，当然，这也是建立在我们的标签是编码成从0开始的状态形式"""
    predict_class = np.argmax(result_P)
    return predict_class




if __name__=='__main__':
    """确实是不好找合适的数据，自己瞎构造吧"""
    X0 = np.array([[0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [2, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 1, 1],
                  [1, 1, 0, 1, 1, 1],
                  [1, 1, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 0],
                  [0, 2, 2, 0, 2, 1],
                  [2, 2, 2, 2, 2, 0],
                  [2, 0, 0, 2, 2, 1],
                  [0, 1, 0, 1, 0, 0],
                  [2, 1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 1, 1],
                  [2, 0, 0, 2, 2, 0],
                  [0, 0, 1, 1, 1, 0]
                  ])
    np.random.seed(5)  #为了实验的可重复性，我把随机种子固定住
    X1 = np.random.random((17, 4)) * 10
    X = np.c_[X0,X1]   # 把 X0，X1 连接起来
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


    is_discrete_feature = np.array([1,1,1,1,1,1,0,0,0,0],dtype=bool)  # 重要，你需要指定那些是离散属性，那些是连续属性，其实可以写程序判断，但是太麻烦（比如去重后超过样本数量一半的就是连续），但是太麻烦了，不如提前指定
    P = compute_class_probability(y)                                  # 计算类别矩阵
    feature_probability_dict = compute_feature_probability_dict(X, y,is_discrete_feature)  # 计算每个属性的类条件概率矩阵，存成字典。连续的就算 均值和方差

    X_test = np.array([2, 1, 1, 0, 2, 0, 4.8, 6.9, 5.67, 8.42])    # 0 类
    # X_test = np.array([1, 0, 0, 0, 0, 0, 4.8, 6.9, 5.67, 8.42])    # 1 类
    predict_class = predict(X_test,P,feature_probability_dict,is_discrete_feature)



