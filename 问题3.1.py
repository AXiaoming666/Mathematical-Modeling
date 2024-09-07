# 导入必要的库
from sklearn.cluster import KMeans  # 导入KMeans聚类算法
from sklearn.metrics import silhouette_score  # 导入轮廓系数评估函数
import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

# 生成合成数据
np.random.seed(0)  # 设置随机种子，确保结果可重复
X = np.random.rand(100, 2)  # 生成100个样本，每个样本2个特征的数据集

# 创建并训练KMeans模型
kmeans = KMeans(n_clusters=2)  # 初始化KMeans模型，设置聚类中心数量为2
kmeans.fit(X)  # 训练模型

# 预测聚类结果
y_pred = kmeans.predict(X)  # 对数据集X进行聚类预测

# 使用轮廓系数评估聚类效果
silhouette_avg = silhouette_score(X, y_pred)  # 计算轮廓系数
print("轮廓系数:", silhouette_avg)  # 打印轮廓系数

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.6)  # 绘制样本点，根据聚类结果着色
centers = kmeans.cluster_centers_  # 获取聚类中心
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8)  # 绘制聚类中心点
plt.title("KMeans聚类")  # 设置图表标题
plt.show()  # 显示图表
