import numpy as np
import pandas as pd
from scipy import linalg


df = pd.read_excel("PCA表格.xlsx", usecols='B:G')     # 读取Excel表格B到G列（去掉标题）;根据实际情况修改
print("表格中数据为:")
print(df)


x = df.to_numpy()
print("建立矩阵x:")
print(x)


X = (x - np.mean(x, axis=0)/np.std(x, ddof=1, axis=0))      # 标准化矩阵
print("标准化后矩阵X:")
print(X)


R = np.cov(X.T)        # 计算协方差矩阵
print("协方差矩阵R:")
print(R)


eigenvalues, eigenvectors = linalg.eigh(R)      # 计算特征值与特征向量
eigenvalues = eigenvalues[::-1]     # 降序排列
eigenvectors = eigenvectors[:, ::-1]


contribution_rate = eigenvalues/sum(eigenvalues)        # 计算贡献率与总贡献率
sum_contribution_rate = np.cumsum(contribution_rate)


print("特征值为:{}".format(eigenvalues))
print("特征向量为:{}".format(eigenvectors))
print("贡献率为:{}".format(contribution_rate))
print("总贡献率为:{}".format(sum_contribution_rate))
