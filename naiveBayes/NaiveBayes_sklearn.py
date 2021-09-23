import numpy as np
from sklearn.naive_bayes import GaussianNB


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
    np.random.seed(5)  # 为了实验的可重复性，我把随机种子固定住
    X1 = np.random.random((17, 4)) * 10
    X = np.c_[X0,X1]   # 把 X0，X1 连接起来
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    clf = GaussianNB()    # 构建一个分类器
    clf.fit(X, y)         # 拟合数据

    # X_test = np.array([2, 1, 1, 0, 2, 0, 4.8, 6.9, 5.67, 8.42]).reshape(1, -1)    # 0 类
    X_test = np.array([1, 0, 0, 0, 0, 0, 4.8, 6.9, 5.67, 8.42]).reshape(1, -1)    # 1 类

    predict_class = clf.predict(X_test)
    predict_class_probability = clf.predict_proba(X_test)
