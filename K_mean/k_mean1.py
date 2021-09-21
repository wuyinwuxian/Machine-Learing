import numpy as np
import matplotlib.pyplot as plt


"""
函数说明:计算两个样本的距离（欧氏距离）
Parameters:
	vector1 - 样本向量1
    vector2 - 样本向量2
Returns:
    两个样本的欧式距离，是一个一个标量
"""
def euclDistance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))


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
    return np.mat(centroids)



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
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterChanged = True
    centroids = initCentroids(dataSet, k)  # 随机初始化几个簇中心

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

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