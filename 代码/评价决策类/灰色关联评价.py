# 适用于不确定母序列
import numpy as np


# 格式为[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
A = np.array(eval(input("请输入初始矩阵:(注意需经过正向化！)")))
Mean = np.mean(A, axis=0)
A_norm = A/Mean     # 预处理后矩阵
print("预处理后矩阵为:{}".format(A_norm))
# 构造母序列与子序列
Y = np.max(A_norm, axis=0)      # 用最大值构造
X = A_norm
# 计算得分
absX0_Xi = np.abs(X - np.tile(Y.reshape(-1, 1), reps=(1, X.shape[1])))      # 计算|X0-Xi|矩阵；这里X0为Y
a = np.min(absX0_Xi)        # 计算a
b = np.max(absX0_Xi)        # 计算b
rho = 0.5       # 分辨系数
gamma = (a + rho * b)/(absX0_Xi + rho * b)      # 关联系数
weight = np.mean(gamma, axis=0)/np.sum(np.mean(gamma, axis=0))      # 权重
score = np.sum(X * np.tile(weight, reps=(X.shape[0], 1)), axis=1)       # 得分
S = score/np.sum(score)     # 归一化
sorted_S = np.sort(S)[::-1]        # 降序排序
index = np.argsort(S)[::-1]        # 索引
print("归一化后得分及索引降序排序:")
print(sorted_S)
print(index)
