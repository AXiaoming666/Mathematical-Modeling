import numpy as np
from scipy.optimize import linprog


# 输入基本数据(缺省输入None)
C = np.array(eval(input("输入价值向量C:")))
print("C:\n{}".format(C))
A_ub = np.array(eval(input("输入不等式约束参数矩阵:")))
print("A_ub:\n{}".format(A_ub))
B_ub = np.array(eval(input("输入不等式约束参数向量:")))
print("B_ub:\n{}".format(B_ub))
A_eq = np.array(eval(input("输入等式约束参数矩阵:")))
print("A_eq:\n{}".format(A_eq))
B_eq = np.array(eval(input("输入等式约束参数向量:")))
print("B_eq:\n{}".format(B_eq))
X = np.array(eval(input("输入决策变量xi的取值范围矩阵:(正无穷为None)")))      # eg: [[0, None],[0, None],[0, 7]]
print("X:\n{}".format(X))


# 转化为标准形式
if input("是最大化问题吗？(T/F)") == 'T':     # 最大化转为最小化
    C = -C
    print("C:\n{}".format(C))
for i in range(A_ub.shape[0]):     # 大于等于转小于等于
    if input("不等式约束参数矩阵第{}行是大于等于吗？(T/F)".format(i+1)) == 'T':
        A_ub[i] = -A_ub[i]
        B_ub[i] = -B_ub[i]
print("A_ub:\n{}".format(A_ub))
print("B_ub:\n{}".format(B_ub))
for i in range(A_eq.shape[0]):
    if input("等式约束参数矩阵第{}行是大于等于吗？(T/F)".format(i+1)) == 'T':
        A_eq[i] = -A_eq[i]
        B_eq[i] = -B_eq[i]
print("A_eq:\n{}".format(A_eq))
print("B_eq:\n{}".format(B_eq))


# 运算并输出结果
resp = linprog(C, A_ub, B_ub, A_eq, B_eq, bounds=X, method='highs')     # fun:最优值；x:此时X取值；nit:迭代次数
print(resp)
