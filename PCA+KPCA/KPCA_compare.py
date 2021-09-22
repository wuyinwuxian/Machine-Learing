import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.decomposition import KernelPCA

"""
函数说明:读取 txt 数据得到 特征矩阵和标签矩阵
Parameters:
	file_path   - 文件所在路径,可以是绝对路径也可以是相对路径
Returns:
	features    - 特征矩阵
	labels      - 标签矩阵
"""
def Load_data(file_path):
    fr = open(file_path)  # 打开数据集
    features = []
    labels = []
    for line in fr.readlines():
        currLine = line.strip().split()
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        features.append(lineArr)
        labels.append(float(currLine[-1]))
    return np.array(features), np.array(labels)

class myKernelPCA:
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


trian_file_path = 'swissroll.txt'
trian_features, trian_labels = Load_data(trian_file_path)

"""绘制原始数据"""
fig = plt.figure()                        # 设置画布fig和绘画区域ax
ax = fig.add_subplot(projection="3d")
label = ["blue", "orange", "green","red"]
for i in range(2):
    ax.scatter(trian_features[i*400:(i+1)*400, 0], trian_features[i*400:(i+1)*400, 1],trian_features[i*400:(i+1)*400, 2], label=label[i], c=label[i])
ax.legend()

# """利用自己写的K_pca 进行降维"""
# kp = myKernelPCA(1, 3)
# z = kp.fit(trian_features[0:800,:])

"""建立目标维度为3的RBF模型"""
scikit_kpca = KernelPCA(n_components=3, kernel='rbf', gamma=15)
z = scikit_kpca.fit_transform(trian_features)

"""绘制降维后的数据"""
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection="3d")
for i in range(2):
    ax1.scatter(z[i*400:(i+1)*400, 0],z[i*400:(i+1)*400, 1],z[i*400:(i+1)*400, 2], label=label[i], c=label[i])
ax.legend()


# """利用自己写的K_pca 进行降维"""
# kp = myKernelPCA(1, 2)
# z = kp.fit(trian_features[0:800,:])

"""建立目标维度为2的RBF模型"""
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
z = scikit_kpca.fit_transform(trian_features)


"""绘制降维后的数据"""
fig2 = plt.figure()
ax2 = fig2.add_subplot()
for i in range(2):
    ax2.scatter(z[i*400:(i+1)*400, 0],z[i*400:(i+1)*400, 1], label=label[i], c=label[i])
ax.legend()


plt.show()

"""
可以看到在这个瑞士卷数据集上，我自己写的K_PCA 也好，sklearn集成的K_PCA也好，降维效果都不行，
或许流形学习可以？还没试过
"""