from scipy.optimize import minimize
import numpy as np
import math


# 定义基本数据
obj_fun = lambda x: x[0] ** 2 + x[1] ** 2 + x[2] **2 + 8    # 目标函数
cons = ({'type': 'ineq', 'fun': lambda x: x[0] ** 2 - x[1] + x[2] ** 2},
        {'type': 'ineq', 'fun': lambda x: 20 - x[0] - x[1] ** 2 - x[2] ** 3},
        {'type': 'eq', 'fun': lambda x: 2 - x[0] - x[1] ** 2},
        {'type': 'eq', 'fun': lambda x: x[1] + 2 * x[2] ** 2 - 3})    # 约束条件('eq'表示等于0，'ineq'表示大于等于0)
bounds = ((0, 2), (0, None), (0, None))    # 变量的上下界,由等式约束条件产生的约束范围要补充


# 蒙特卡罗法估计最优解
N = 100000    # 蒙特卡罗模拟次数
n = len(bounds)    # 变量个数
x_min = np.zeros(n)    # 最优解
obj_fun_min = np.inf    # 目标函数的最小值
for i in range(N):
    x = np.zeros(n)
    # 随机生成变量
    while True:
        for j in range(1):    # 循环几次决定于等式约束条件的秩，示例情况下决策变量只有一个自由度
            if(bounds[j][0] == None and bounds[j][1] == None):
                x[j] = np.random.exponential(1.0) - np.random.exponential(1.0)
            elif(bounds[j][0] == None):
                x[j] = bounds[j][1] - np.random.exponential(1.0)
            elif(bounds[j][1] == None):
                x[j] = np.random.exponential(1.0) + bounds[j][0]
            else:
                x[j] = np.random.uniform(bounds[j][0], bounds[j][1])
        # 根据等式约束条件补足变量
        x[1] = math.pow(2 - x[0], 0.5)
        x[2] = math.pow((3 - x[1]) / 2, 0.5)
        # 判断是否满足不等式约束条件
        if all(((cons[k]['type'] == 'ineq' and cons[k]['fun'](x) >= 0) or (cons[k]['type'] == 'eq')) for k in range(len(cons))):
            break
    obj_fun_val = obj_fun(x)    # 计算目标函数的值
    if obj_fun_val < obj_fun_min:
        obj_fun_min = obj_fun_val
        x_min = x
print('蒙特卡罗法估计最优解:', x_min)
print('蒙特卡罗法估计的目标函数的最小值:', obj_fun_min)


# 数学求解
""" minimize函数method参数
1. 'Nelder-Mead'（也称为单纯形法）:一种启发式算法，适用于无约束问题，不需要梯度信息。
2. 'Powell':Powell方法是一种共轭方向方法，适用于无约束问题，不需要梯度信息。
3. 'CG'（共轭梯度法）:需要目标函数的一阶导数，适用于无约束问题。
4. 'BFGS':BFGS方法是一种拟牛顿方法，需要目标函数的一阶导数，适用于无约束问题。
5. 'Newton-CG':牛顿共轭梯度法，需要目标函数的一阶和二阶导数，适用于无约束问题。
6. 'L-BFGS-B':限制内存的BFGS方法，需要目标函数的一阶导数，适用于有边界约束的问题。
7. 'TNC':Truncated Newton（截断牛顿）算法，需要目标函数的一阶导数，适用于有边界约束的问题。
8. 'COBYLA':约束优化方法，不需要梯度信息，适用于有边界约束的问题。
9. 'SLSQP':序列二次规划方法，需要目标函数的一阶导数，适用于有边界和等式/不等式约束的问题。
10. 'trust-constr':信赖域约束优化方法，是SciPy中较新的方法，需要目标函数的一阶导数，适用于有边界和等式/不等式约束的问题。"""
res = minimize(obj_fun, x_min, method = 'trust-constr', bounds = bounds, constraints = cons)
if res.success:
    print('数学求解的最优解:', res.x)  # 最优解
    print('数学求解的目标函数的最小值:', res.fun)  # 目标函数的最小值
else:
    print('数学求解失败！')