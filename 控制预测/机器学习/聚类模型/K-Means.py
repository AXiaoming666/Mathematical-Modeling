from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import X, y
import matplotlib.pyplot as plt

# 假设 X 是特征集，y 是目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建回归模型
log_reg = KMeans(n_clusters=2)

# 训练模型
log_reg.fit(X_train, y_train)

# 预测测试集
y_pred = log_reg.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

# 可视化
plt.scatter(X_train[:, 0], X_train[:, 1], c=log_reg.labels_, cmap='viridis')
centers = log_reg.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()