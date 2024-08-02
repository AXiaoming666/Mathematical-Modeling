from scipy.optimize import linprog
import numpy as np


# 定义基本数据
C = np.array([2800,4500,6000,7300,2800,4500,6000,7300,2800,4500,6000,7300,2800,4500,6000,7300])    # 价值向量
A_gq = np.array([[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                 [0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0],
                 [0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1]])    # 不等式约束条件的变量系数矩阵(大于等于)
b_gq = np.array([15,10,20,12])    # 不等式约束条件的常数项向量(大于等于)
A_lq = np.array([])    # 不等式约束条件的变量系数矩阵(小于等于)
b_lq = np.array([])    # 不等式约束条件的常数项向量(小于等于)
# 等式约束条件的变量系数矩阵与常数项向量不能为空，若无约束，全零
A_eq = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])    # 等式约束条件的变量系数矩阵
b_eq = np.array([0])    # 等式约束条件的常数项向量
X = np.array([[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None],[0,None]])    # 决策变量xi的取值范围矩阵(正无穷为None)
Mxx = 0    # 最大化问题取1，最小化问题取0


# 定义线性规划函数
def LP(C, A_gq, b_gq, A_lq, b_lq, A_eq, b_eq, X, Mxx):
    # 统一为小于等于不等式的形式
    if np.size(A_lq) & np.size(A_gq):
            A_ub = np.concatenate((A_lq,-A_gq), axis=0)
            b_ub = np.concatenate((b_lq,-b_gq), axis=0)
    else:
        if np.size(A_lq) == 0:
            A_ub = -A_gq
            b_ub = -b_gq
        else:
            A_ub = A_lq
            b_ub = b_lq


    # 若为最大化问题，价值函数两边取负号，统一为最小化问题;若为最小化问题，直接代入linprog函数
    if Mxx:    #最大化问题
        result = linprog(-C, A_ub, b_ub, A_eq, b_eq, X)
        result.fun = -result.fun
    else:    #最小化问题
        result = linprog(C, A_ub, b_ub, A_eq, b_eq, X)
    return result


result = LP(C, A_gq, b_gq, A_lq, b_lq, A_eq, b_eq, X, Mxx)
print("x =", result.x)
print("F(x) =", result.fun)