import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


# 你的数据
x_data = np.array([0.0, 0.20408163265306123, 0.40816326530612246, 0.6122448979591837, 0.8163265306122449, 1.0204081632653061])
y_data = np.array([1.0, 1.2263982635139905, 1.5040527007501314, 1.8445676204334889, 2.262174526633764, 2.774326911229232])


# 创建模型实例并拟合
model = sm.OLS(np.log(y_data), sm.add_constant(x_data))
result = model.fit()

# result.params包含拟合得到的参数
print(result.summary())

# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.scatter(x_data, y_data, label='实际值')      # 散点图
x = np.linspace(0, 1, 100)
plt.plot(x, np.exp(result.params[0] + result.params[1] * x), color='red', label='预测值')  # 预测值
plt.legend()        # 显示图例，即每条线对应label中的内容
plt.show()      # 显示图形