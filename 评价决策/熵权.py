import numpy as np


def calclog(p):  # 计算对数
    nn = len(p)
    lnp = np.zeros(nn)
    for k in range(nn):
        if p[k] == 0:
            lnp[k] = 0
        else:
            lnp[k] = np.log(p[k])
    return lnp


print("参评对象数目")
n = int(input())
print("输入指标数:")
m = int(input())
print("输入指标矩阵X:(X应为正向化后矩阵)")
X = np.zeros(shape=(n, m))
for i in range(n):
    X[i] = input().split(' ')
    X[i] = list(map(float, X[i]))
print("矩阵X为:\n{}".format(X))
# 标准化处理X
Z = X / np.sqrt(np.sum(X * X, axis=0))
print("标准化后矩阵Z为\n{}".format(Z))
# 计算信息效用值
D = np.zeros(m)
for i in range(m):
    x = Z[:, i]
    p = x/np.sum(x)
    e = -np.sum(p*calclog(p))/np.log(n)
    D[i] = 1-e
# 计算熵权
W = D/np.sum(D)
print("权重W:{}".format(W))
