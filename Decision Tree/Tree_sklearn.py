import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from random import randint


'''  预测白葡萄酒  '''
print("预测白葡萄酒")

'''导入数据'''
white = pd.read_csv("winequality-white.csv")

'''检查数据是否需要处理'''
# print(white.dtypes)                        #查看特征的数据类型，都为float型，不需要进行数据类型处理。
# print(white.isnull().any())                #发现没有缺失值，所以不需要进行缺失值处理。
# print(white['quality'].value_counts())     #数据分布不平衡，选择过采样，用imbalanced-learn进行处理

'''数据是处理'''
white_x = white.iloc[:,:-1].values         # iloc方法根据位置选择，即选择所有行，所有列去掉右数第一列
white_y = white['quality'].values
ros = RandomOverSampler()                  # 构造采样方法，来使得样本均衡
balanced_white_x,balanced_white_y = ros.fit_resample(white_x,white_y)
# print(pd.DataFrame(y1)[0].value_counts().sort_index())   #显示采样结果

'''构建决策树'''
DTC1 = DecisionTreeClassifier(max_depth = 15, min_samples_leaf = 4)               # 初始化  可以指定 criterion='gini' or “entropy” 默认‘gini'
DTC1.fit(balanced_white_x,balanced_white_y)   # 训练 or 拟合

'''评价模型'''
winescore1 = cross_val_score(DTC1,balanced_white_x,balanced_white_y,cv=5,scoring='accuracy')  # 交叉验证 5折，最后取均值
print(np.mean(winescore1))

'''绘出所建立的决策树'''
# tree.plot_tree(DTC1)

'''预测'''
index = randint(0,white_x.shape[0])
test  = white_x[index,:].reshape(1, -1)
class_test = DTC1.predict(test)
print("样本序号：" + str(index) + '\n'  + "真实标签：" + str(white_y[index]) + '\n' + "预测标签：" + str(class_test[0]))



'''  预测红葡萄酒  '''
print("\n预测红葡萄酒")

'''导入数据'''
red   = pd.read_csv("winequality-red.csv")

'''数据不平衡处理'''
red_x = red.iloc[:,:-1].values
red_y = red['quality'].values
ros2  = RandomOverSampler()
balanced_red_x,balanced_red_y = ros.fit_resample(red_x,red_y)

'''构建决策树'''
DTC2 = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 5)
DTC2.fit(balanced_red_x,balanced_red_y)

'''评价模型'''
winescore2 = cross_val_score(DTC2,balanced_red_x,balanced_red_y,cv=5,scoring='accuracy')
print(np.mean(winescore2))

'''预测'''
index = randint(0,red_x.shape[0])
test  = white_x[index,:].reshape(1, -1)
class_test = DTC1.predict(test)
print("样本序号：" + str(index) + '\n'  + "真实标签：" + str(white_y[index]) + '\n' + "预测标签：" + str(class_test[0]))