import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

X = np.genfromtxt('planting_2023_matrix.csv', delimiter=',')
C = np.genfromtxt('planting_cost_matrix.csv', delimiter=',')
S = np.genfromtxt('planting_salesprice_matrix.csv', delimiter=',')
Y = np.genfromtxt('planting_yield_matrix.csv', delimiter=',')

anticipate = (Y * X).sum(axis=1)
sales_price = S
planting_cost = np.zeros((59, 54))
""" for i in range(59):
    for j in range(54):
        if Y[i, j] != 0:
            planting_cost[i, j] = C[i, j] / Y[i, j] """
planting_cost = np.sum(C, axis=1) / np.count_nonzero(C, axis=1)
print(planting_cost)


df = pd.DataFrame({'anticipate': anticipate, 'sales_price': sales_price, 'planting_cost': planting_cost})

print(df)

# 生成热力图
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(method='pearson'), annot=True, cmap='viridis')
plt.title('热力图')
plt.show()