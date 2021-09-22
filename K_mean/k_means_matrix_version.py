import numpy as np
import matplotlib.pyplot as plt

"""
函数说明:计算样本和簇中心距离矩阵
Parameters:
	dataSet   - 样本矩阵  （样本*特征）
    centroids - 簇中心矩阵 （k*特征）
Returns:
    distance  - 样本和簇中心距离矩阵
"""
def Distance(dataSet, centroids):
    numSamples = dataSet.shape[0]
    k = centroids.shape[0]
    distance = np.zeros((numSamples, k))
    for i in range(k):
        temp = np.sqrt(np.sum(np.power(dataSet - centroids[i], 2,),axis=1))  # 每个样本和第 i 个中心求距离
        distance[:,i] = temp.reshape(1,-1)
    return distance

"""
函数说明:随机初始化 k 个聚类中心，就是随机从数据集中选 k 个作为中心
Parameters:
	dataSet - 样本矩阵
    k       - 你指定的聚类个数
Returns:
    np.mat(centroids) - k 个聚类中心中心的构成矩阵
"""
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))   # 随机选一个 index 下标出来，作为中心
        centroids[i, :] = dataSet[index, :]
    return centroids


"""
函数说明:根据数据矩阵，和样本的状态矩阵更新中心
Parameters:
	dataSet        - 数据矩阵
    clusterAssment - 每个样本它属于的簇和它到这个簇中心距离构成的矩阵
    k              - 聚类的类别数
Returns:
    centroids      - 新的中心的坐标矩阵
"""
def update_centroids(dataSet, clusterAssment,k):
    centroids = np.zeros((k, dataSet.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(dataSet[clusterAssment[:,0]==i],axis=0)
    return centroids


"""
函数说明:结果可视化,二维的情况
Parameters:
	dataSet        - 数据矩阵
    k              - 聚类的类别数
Returns:
    centroids      - 聚类中心的坐标矩阵
    clusterAssment - 每个样本它属于的簇和它到这个簇中心距离构成的矩阵
"""
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = np.zeros((numSamples, 2))
    centroids = initCentroids(dataSet, k)  # 随机初始化几个簇中心

    for i in range(100):
        distance = Distance(dataSet, centroids)                                               # 计算距离矩阵
        samples_class = np.argmin(distance, axis=1)                                           # 根据最小距离判断每个样本属于哪一个簇
        clusterAssment[:,0] = samples_class
        clusterAssment[:,1] = distance[range(dataSet.shape[0]), samples_class]                # 把每个样本到他所属簇中心的距离保存
        centroids = update_centroids(dataSet, clusterAssment,k)
    print('Congratulations, cluster complete!')

    return centroids, clusterAssment


"""
函数说明:结果可视化,二维的情况
Parameters:
	dataSet        - 数据矩阵
    k              - 聚类的类别数
    centroids      - 聚类中心的坐标矩阵
    clusterAssment - 每个样本它属于的簇和它到这个簇中心距离构成的矩阵
Returns:
    无
"""
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  # 预定了这么多种标记颜色
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

    # 开始绘图
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']   # 同理，这是中心的标记，
    # 把中心画出来
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


if __name__ == '__main__':
    ## step 1: 导入数据 指定类别数
    dataSet = np.mat(np.loadtxt('./test.txt'))
    k = 4

    ## step 2: 聚类，返回聚类中心和，每个样本的类别
    centroids, clusterAssment = kmeans(dataSet, k)

    ## step 3: 显示结果
    showCluster(dataSet, k, centroids, clusterAssment)