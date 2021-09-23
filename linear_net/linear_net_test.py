import random
import torch
from d2l import torch as d2l
""" d2l 是《跟着李沐学深度学习》，大神他们写的一个包，里面放了一些他们的东西， b站地址  https://space.bilibili.com/1567748478/channel/detail?cid=175509"""

"""
函数描述：生成数据集
参数：
    w            - 权重
    b            - 偏置
    num_examples - 产生的样本个数
返回值：
    X - 样本数据矩阵
    y - 标签矩阵
"""
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

"""
函数描述：对数据进行随机分批，注意当样本数不能整除的时候，最后一批可能不满 batch_size 个样本
参数：
    batch_size - 批大小，即每次随机选batch_size个样本返回
    features   - 特征矩阵
    labels     - 标签矩阵
返回值：
    features[batch_indices] - 随机的样本batch_size个样本数据矩阵
    labels[batch_indices]   - 对应的标签矩阵
"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)                       # 把样本序号打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])   # min 的作用是防止最后一次不够 batch_size 个样本了
        yield features[batch_indices], labels[batch_indices]

"""
函数描述：线性回归模型。
参数：
    w - 权重
    b - 偏置
    X - 样本数据矩阵
返回值：
    y - 标签矩阵 y = wx+b
"""
def linreg(X, w, b):
    return torch.matmul(X, w) + b    # matmul是矩阵乘法  (m×k) * (k×n) = m×n

"""
函数描述：计算均方损失。预测值和真实值的均方差
参数：
    y_hat - 模型预测出来的标签矩阵,由于是单输出回归，所以其实是一个m*1的， m样本数
    y     - 真实标签矩阵 m*1的， m样本数
返回值
   均方损失 （y_hat - y）
"""
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

"""
函数描述：随机梯度下降算法，更新的参数直接就反应在 param 里面了,似乎是pytoch的一种特点？我其实觉得有点奇怪，在函数内部的变化会影响到实参吗，难道传过来的是一个引用?
        我又看了下，应该是由于 param 是可变对象，而且 += 操作符对可变对象是原地修改的，然后可能还涉及到变量作用域的相关知识，暂时我还没高得很明白
        https://discuss.d2l.ai/t/topic/1778/3 ，看下这个链接的讨论，看能否有点收获
参数：
    params     - 要优化的参数
    lr         - 学习率，步长
    batch_size - 批大小
返回值
   无
"""
def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            # param.grad 可以这么理解 params是一个对象数组，包含了两个对象 w、b。grad是对象的一个属性，通过操作符.来获取对应的属性值
            param -= lr * param.grad / batch_size
            param.grad.zero_()   # 必须把梯度清0 ，不然会一直累计


if __name__=="__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)   # 生成 1000 个数据

    """看看数据集长啥样"""
    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
    # d2l.plt.show()

    batch_size = 10
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)    # requires_grad=True 代表我要梯度信息
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linreg          # 给函数起别名，方便记忆，也是方便简化代码
    loss = squared_loss   # 给函数起别名，方便记忆

    """开始训练"""
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)    # `X`和`y`的小批量损失
            # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，并以此计算关于[`w`, `b`]的梯度，.backward() 会自动求关于 w,b的梯度,
            # 放到w，b的 grad 属性中，这个我其实不太明白，感觉差点意思，他咋实现的呢这是，这东西得看看 pytorch 内部咋构造的，我没往下深究了
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')