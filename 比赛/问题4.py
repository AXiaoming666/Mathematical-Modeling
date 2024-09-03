import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 读取数据
data = pd.read_excel('附件.xlsx', header=None)

# 绘制3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 生成X和Y的网格
X = np.arange(0, 4.02, 0.02)
Y = np.arange(0, 5.02, 0.02)
X, Y = np.meshgrid(X, Y)

# 提取Z数据，确保索引正确
Z = data.values[2:253, 2:203]
sc = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c='r', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())
color_bar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

# 显示图形
plt.show()