# coding: utf-8

from src.scratch.common.functions import *
from src.scratch.common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化权重
        :param input_size: 输入层的神经元数
        :param hidden_size: 隐藏层的神经元数
        :param output_size: 输出层的神经元数
        :param weight_init_std:
        """
        # params 保存神经网络的参数
        # W1 表示第1层的权重，b1 表示第1层的偏置
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size),  # 第一层的权重
                       'b1': np.zeros(hidden_size),  # 第一层的偏置
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size),  # 第二层的权重
                       'b2': np.zeros(output_size)}  # 第二层的偏置

    def predict(self, x):
        """
        进行识别
        :param x: 图像数据
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """
        计算损失函数值
        :param x: 图像数据
        :param t: 监督数值
        :return: 损失值
        """
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """
        计算识别精度
        :param x:
        :param t:
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        """
        计算权重参数的梯度
        :param x:
        :param t:
        :return:
        """
        loss_W = lambda W: self.loss(x, t)

        # 保存梯度
        grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']),
                 'b2': numerical_gradient(loss_W, self.params['b2'])}

        return grads

    def gradient(self, x, t):
        """
        使用误差反向传播计算权重参数的梯度，速度更快。
        :param x:
        :param t:
        :return:
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
assert net.params['W1'].shape == (784, 100)
assert net.params['b1'].shape == (100,)
assert net.params['W2'].shape == (100, 10)
assert net.params['b2'].shape == (10,)

x = np.random.rand(100, 784)  # 伪输入数据
t = np.random.rand(100, 10)  # 伪正解标签
y = net.predict(x)

grads = net.numerical_gradient(x, t)
