import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler

"""为了方便就直接用sklearn里面的决策树了啊，当然我觉得把我们之前实现的决策树导入进来也没毛病，我们理解下思想"""

def random_forest(classifier_number,data,label):
    classifier_lists = []
    samples = data.shape[0]
    for i in range(classifier_number):
        sample_indices = np.random.randint(0,samples,samples)
        X = data[sample_indices,:]
        Y = label[sample_indices]
        DTC = DecisionTreeClassifier(max_depth=20)
        # DTC = DecisionTreeClassifier()     # 不指定最大深度的话可以到99.9%
        DTC.fit(X,Y)
        classifier_lists.append(DTC)
    return classifier_lists

def predict(data,classifier_lists,y):
    classifier_number = len(classifier_lists)
    samples = data.shape[0]
    res_array = np.ones((samples, classifier_number),dtype=int)

    for i in range(classifier_number):
        pre = classifier_lists[i].predict(data)    # 第i个分类器的预测结果
        res_array[:,i] = pre.reshape((samples,))   # 加到第i列
        print("第", i ,'棵树准确率:' , sum(pre == y) * 1. / len(y))

    predict_result = np.ones((samples,),dtype=int)*-1   # 全部初始化为-1
    for i in range(samples):
        predict_result[i] = np.argmax(np.bincount(res_array[i,:]))

    return predict_result

if __name__ == '__main__':
    '''导入数据'''
    white = pd.read_csv("../Decision Tree/winequality-white.csv")

    '''数据不平衡处理'''
    white_x = white.iloc[:, :-1].values  # iloc方法根据位置选择，即选择所有行，所有列去掉右数第一列
    white_y = white['quality'].values
    ros = RandomOverSampler()  # 构造采样方法，来使得样本均衡
    balanced_white_x, balanced_white_y = ros.fit_resample(white_x, white_y)

    classifier_number = 30   # 测试10个决策树的随机森林
    trees = random_forest(classifier_number, balanced_white_x, balanced_white_y)
    y_pred = predict(white_x,trees,white_y)
    print('准确率:', sum(y_pred == white_y) * 1. / len(white_y))

    """集成学习确实要比单个决策器好一些，实验结果可以看到，大概提高了10%"""









