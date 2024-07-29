# 适用于母序列已知
import numpy as np


# 格式为[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
A = np.array(eval(input("请输入初始矩阵:")))
Mean = np.mean(A, axis=0)
A_norm = A/Mean     # 预处理后矩阵
print("预处理后矩阵为:{}".format(A_norm))
Y = A_norm[:, 0]    # 母序列（第一列）
X = A_norm[:, 1:]   # 子序列（后面几列）
absX0_Xi = np.abs(X - np.tile(Y.reshape(-1, 1), reps=(1, X.shape[1])))      # 计算|X0-Xi|矩阵；这里X0为Y
a = np.min(absX0_Xi)        # 计算a
b = np.max(absX0_Xi)        # 计算b
rho = 0.5       # 分辨系数
gamma = (a + rho * b)/(absX0_Xi + rho * b)      # 关联系数
print("子序列各指标关联度为:")
print(np.mean(gamma, axis=0))
