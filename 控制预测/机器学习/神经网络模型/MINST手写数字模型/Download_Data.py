import requests

# MNIST数据集的URL
url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

# 下载数据集
response = requests.get(url)
with open('控制预测\机器学习\神经网络模型\MINST手写数字模型\mnist.npz', 'wb') as file:
    file.write(response.content)