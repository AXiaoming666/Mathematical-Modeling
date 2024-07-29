import pulp as pp
import numpy as np


# 输入基本数据(缺省输入None)
C = np.array(eval(input("输入价值向量C:")))
print("C:\n{}".format(C))
A_gq = np.array(eval(input("输入不等式约束参数矩阵(大于等于):")))
print("A_gq:\n{}".format(A_gq))
B_gq = np.array(eval(input("输入不等式约束参数向量(大于等于):")))
print("B_gq:\n{}".format(B_gq))
A_lq = np.array(eval(input("输入不等式约束参数矩阵(小于等于):")))
print("A_lq:\n{}".format(A_lq))
B_lq = np.array(eval(input("输入不等式约束参数向量(小于等于):")))
print("B_lq:\n{}".format(B_lq))
A_eq = np.array(eval(input("输入等式约束参数矩阵:")))
print("A_eq:\n{}".format(A_eq))
B_eq = np.array(eval(input("输入等式约束参数向量:")))
print("B_eq:\n{}".format(B_eq))
X = np.array(eval(input("输入决策变量xi的取值范围矩阵:(正无穷为None)")))      # eg: [[0, None],[0, None],[0, 7]]
print("X:\n{}".format(X))


# 建立模型
if input("为最大化问题吗？(T/F)") == 'T':
    prob = pp.LpProblem('max', sense=pp.LpMaximize)
    x = [pp.LpVariable(f'x{i}', lowBound=X[i-1][0], upBound=X[i-1][1], cat='Integer')for i in range(X.shape[0])]        # 参数integer为整数规划，Continuous为连续变量，Binary为0-1规划
    prob += pp.lpDot(C, x)
    for i in range(A_gq.shape[0]):
        prob += (pp.lpDot(A_gq[i], x) >= B_gq[i])
    for i in range(A_lq.shape[0]):
        prob += (pp.lpDot(A_lq[i], x) <= B_lq[i])
    for i in range(A_eq.shape[0]):
        prob += (pp.lpDot(A_eq[i], x) == B_eq[i])
else:
    prob = pp.LpProblem('min', sense=pp.LpMinimize)
    x = [pp.LpVariable(f'x{i}', lowBound=X[i-1][0], upBound=X[i-1][1], cat='Integer') for i in range(X.shape[0])]
    prob += pp.lpDot(C, x)
    for i in range(A_gq.shape[0]):
        prob += (pp.lpDot(A_gq[i], x) >= B_gq[i])
    for i in range(A_lq.shape[0]):
        prob += (pp.lpDot(A_lq[i], x) <= B_lq[i])
    for i in range(A_eq.shape[0]):
        prob += (pp.lpDot(A_eq[i], x) == B_eq[i])


# 求解并输出
prob.solve()
print("Status:", pp.LpStatus(prob.status))      # 求解状态
for v in prob.variables():
    print(v.name, "=", v.varValue)      # 变量最优值
print("F(x) = ", pp.value(prob.objective))      # 最优解目标函数值
