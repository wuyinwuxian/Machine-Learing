import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


"""
函数说明:结果可视化,二维的情况
Parameters:
	data           - 数据矩阵
    k              - 聚类的类别数
    centroids      - 聚类中心的坐标矩阵
    label_pred     - 聚类后预测的类别矩阵
Returns:
    无
"""
def showCluster(data, k, centroids, label_pred):
    mark = ['r', 'b', 'g', 'k']
    plt.figure()
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], c=mark[label_pred[i]])   # 一个点一个点的画

    # 把中心画出来
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


if __name__ == '__main__':
    ## step 1: 导入数据 指定类别数
    data = np.loadtxt('./test.txt')
    k = 4

    # step 2: 构造一个聚类器
    k_mean = KMeans(n_clusters=k)        # 构造聚类器
    k_mean.fit(data)                     # 聚类

    # step 3: 画图
    label_pred = k_mean.labels_          # 获取聚类标签
    centroids = k_mean.cluster_centers_  # 获取聚类中心
    showCluster(data, k, centroids, label_pred)




