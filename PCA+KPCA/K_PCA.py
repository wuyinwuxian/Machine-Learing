import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.datasets import make_circles,make_moons



class KernelPCA:
    def __init__(self, bandwidth, dim):
        self.bandwidth = bandwidth          # 定义核函数的带宽
        self.dim = dim                      # 定义隐变量的维数

    def fit(self, train_X):
        self.size_train = train_X.shape[0]
        self.train_X = train_X

        dist = squareform(pdist(self.train_X))    # 计算样本的欧氏距离，它默认返回压缩的距离元组，然后利用 squareform 扩展下变成一个对称的方阵，对角线为0
        K = self.rbf_kernel(dist)

        self.one = np.ones((self.size_train, 1), dtype=np.float32)
        O = self.one.dot(self.one.T) / self.size_train
        K_hat = K - O.dot(K) - K.dot(O) + O.dot((K.dot(O)))   # 对 K 去中心化，pca处理去中心化的数据

        Lambda, Q  = np.linalg.eig(K_hat)                     # 计算特征值D 和特征向量 U
        index = np.argsort(Lambda)[::-1]                      # 先排序后反转，变成从大到小，不能指定reverse就很难受
        index_k = index[0:self.dim]                           # 把前k个的索引找到
        return Q[:,index_k]


    def rbf_kernel(self, X):
        return np.exp(-X / (self.bandwidth**2))



"""生成数据"""
np.random.seed(123)   # 设置随机种子
mean_list = ([0, 1, 2], [4, 3, 5], [7, 8, 9])                              # 三组数据的均值
cov = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]], dtype=np.float32)        # 三组数据的方差
cita = [0.5, 0.8, 0.5]                                                     # 用来控制方差大小的参数
sample = list()
for mean, cita in zip(mean_list, cita):
    data = np.random.multivariate_normal(mean, cov=cita * cov, size=100)  # 采样出三组数据
    sample.append(data)

"""设置画布fig和绘画区域ax"""
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

"""绘制原始数据"""
label = ["blue", "orange", "green"]
for P, label in zip(sample, label):
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], label=label, c=label)
ax.legend()


"""利用K_pca 进行降维"""
kp = KernelPCA(1, 2)
data = np.concatenate((sample[0], sample[1], sample[2]), axis=0)      # 将三组数据连接起来
z = kp.fit(data)



"""绘制降维后的数据"""
fig = plt.figure()
a, b, c = np.split(z, 3, axis=0)
latent_list = [a, b, c]
colors = ["blue", "orange", "green"]
for z, color in zip(latent_list, colors):
    plt.scatter(z[:, 0], z[:, 1], c=color)


plt.show()


