import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# 假设我们有以下数据：
# X1, X2，X3 是自变量
# y 是因变量
X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([5, 4, 3, 2, 1])
X3 = np.array([6, 7, 8, 9, 10])
y = np.array([2, 3, 4, 5, 6])

# 将自变量组合成一个二维数组
X = np.column_stack((X1, X2, X3))

# 添加常数项以拟合截距
X = sm.add_constant(X)

# 构建并拟合模型
model = sm.OLS(y, X).fit()

# 输出回归结果摘要
print(model.summary())

# 进行预测
predictions = model.predict(X)

# 如果需要，可以绘制结果（这里需要具体的绘图数据）
plt.scatter(X1, y, color='blue')
plt.plot(X1, predictions, color='red', linewidth=2)
plt.show()