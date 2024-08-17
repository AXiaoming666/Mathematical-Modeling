import numpy as np

# 加载本地保存的数据集
with np.load('控制预测\机器学习\神经网络模型\MINST手写数字模型\mnist.npz') as data:
    X_train = data['x_train']
    y_train = data['y_train']
    X_test = data['x_test']
    y_test = data['y_test']