import pulp as pp
import numpy as np

# 定义基本数据(若无对应约束条件，则置为空数组)
C = np.array([15,13.8,12.5,11,14.3,
              14.5,14,13.2,10.5,15,
              13.8,13,12.8,11.3,14.6,
              14.7,13.6,13,11.6,14])    # 价值向量
A_gq = np.array([])    # 不等式约束条件的变量系数矩阵(大于等于)
b_gq = np.array([])    # 不等式约束条件的常数项向量(大于等于)
A_lq = np.array([[1,1,1,1,1,
                 0,0,0,0,0,
                 0,0,0,0,0,
                 0,0,0,0,0,],
                [0,0,0,0,0,
                 1,1,1,1,1,
                 0,0,0,0,0,
                 0,0,0,0,0,],
                [0,0,0,0,0,
                 0,0,0,0,0,
                 1,1,1,1,1,
                 0,0,0,0,0,],
                [0,0,0,0,0,
                 0,0,0,0,0,
                 0,0,0,0,0,
                 1,1,1,1,1,]])    # 不等式约束条件的变量系数矩阵(小于等于)
b_lq = np.array([2,2,2,2])    # 不等式约束条件的常数项向量(小于等于)
A_eq = np.array([[1,0,0,0,0,
                  1,0,0,0,0,
                  1,0,0,0,0,
                  1,0,0,0,0,],
                 [0,1,0,0,0,
                  0,1,0,0,0,
                  0,1,0,0,0,
                  0,1,0,0,0,],
                 [0,0,1,0,0,
                  0,0,1,0,0,
                  0,0,1,0,0,
                  0,0,1,0,0,],
                 [0,0,0,1,0,
                  0,0,0,1,0,
                  0,0,0,1,0,
                  0,0,0,1,0,],
                 [0,0,0,0,1,
                  0,0,0,0,1,
                  0,0,0,0,1,
                  0,0,0,0,1,]])    # 等式约束条件的变量系数矩阵
b_eq = np.array([1,1,1,1,1])    # 等式约束条件的常数项向量
X_type = np.array([0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0,])    # 决策变量xi的取值类型（0：0-1变量；1：整数变量；2：连续变量）
dict([(0,'Binary'),(1,'Integer'),(2,'Continuous')])    # 决策变量xi的取值类型字典
X = np.array([[0,None],[0,None],[0,None],[0,None],[0,None],
              [0,None],[0,None],[0,None],[0,None],[0,None],
              [0,None],[0,None],[0,None],[0,None],[0,None],
              [0,None],[0,None],[0,None],[0,None],[0,None]])    # 决策变量xi的取值范围矩阵(正无穷为None)
if_max = False    # 最大化问题取True，最小化问题取False


# 建立模型
# 判断优化问题类型
if if_max:
    prob = pp.LpProblem('max', sense=pp.LpMaximize)
else:
    prob = pp.LpProblem('min', sense=pp.LpMinimize)
x = [pp.LpVariable(f'x{i}', lowBound = X[i][0], upBound = X[i][1], cat = dict[X_type[i]])for i in range(X.shape[0])]    # 定义决策变量
prob += pp.lpDot(C, x)    # 定义目标函数
# 定义约束条件
for i in range(A_gq.shape[0]):
    prob += (pp.lpDot(A_gq[i], x) >= b_gq[i])
for i in range(A_lq.shape[0]):
    prob += (pp.lpDot(A_lq[i], x) <= b_lq[i])
for i in range(A_eq.shape[0]):
    prob += (pp.lpDot(A_eq[i], x) == b_eq[i])


# 求解并输出
prob.solve()
print("Status:", pp.LpStatus[prob.status])      # 求解状态
for v in prob.variables():
    print(v.name, "=", v.varValue)      # 变量最优值
print("F(x) = ", pp.value(prob.objective))      # 最优解目标函数值
