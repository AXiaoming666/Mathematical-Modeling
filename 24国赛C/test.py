import numpy as np

Anti = np.genfromtxt('planting_sales_matrix.csv', delimiter=',')[-4:]
S = np.genfromtxt('planting_salesprice_matrix.csv', delimiter=',')[-4:]

Q0 = Anti
P0 = S
print(Q0)
print(P0)

Q_max = 1.05 * Q0
Q_min = 0.95 * Q0
P_max = 0.99 * P0
P_min = (0.95 ** 7) * P0

b = np.log(Q_max / Q_min) / np.log(P_max / P_min)
a = Q_max * (P_min ** b)

# 打印散点图
import matplotlib.pyplot as plt

X = np.linspace(0.95, 1.05, 1000)
Y = a[0] / (X ** b[0])


plt.plot(X, Y, color='b', label='Fitted')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.legend()
plt.show()