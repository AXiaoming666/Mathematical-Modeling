import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 4, 5, 6, 7])
X = sm.add_constant(x)      # 添加截距(如果有)
model = sm.OLS(y, X, hasconst=1)        # 构建模型并拟合
results = model.fit()


print(results.summary())


# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


predicts = results.predict()      # 模型的预测值

plt.scatter(x, y, label='实际值')      # 散点图
plt.plot(x, predicts, color='red', label='预测值')     # 直线
plt.legend()        # 显示图例，即每条线对应label中的内容
plt.show()      # 显示图形
