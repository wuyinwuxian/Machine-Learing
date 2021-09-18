from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# 获得半月形的数据集
X, y = make_moons(n_samples=100, random_state=123)
plt.figure(0)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.title("原始数据")
plt.tight_layout()


"""建立目标维度为2的RBF模型"""
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)         # 使用KPCA降低数据维度，直接获得投影后的坐标
# 数据可视化
plt.figure(1)
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.title("降维后二维上的可视化")
plt.tight_layout()


"""建立目标维度为1的RBF模型"""
scikit_kpca = KernelPCA(n_components=1, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
# 数据可视化
plt.figure(2)
plt.scatter(X_skernpca[y==0, 0], y[y==0], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], y[y==1], color='blue', marker='o', alpha=0.5)
plt.title("降维后一维上的可视化")
plt.tight_layout()
plt.show()

