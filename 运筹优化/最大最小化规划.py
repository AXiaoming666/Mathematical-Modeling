from scipy.optimize import minimize
import numpy as np
import math


# 定义基本数据
obj_fun = lambda x: max(np.sqrt((x[0] - 10)**2 + (x[1] - 20)**2),
                        np.sqrt((x[0] - 15)**2 + (x[1] - 30)**2),
                        np.sqrt((x[0] - 20)**2 + (x[1] - 25)**2),
                        np.sqrt((x[0] - 25)**2 + (x[1] - 15)**2))    # 目标函数
cons = ({'type': 'ineq', 'fun': lambda x: np.sqrt((x[0] - 10)**2 + (x[1] - 20)**2) - 5},
        {'type': 'ineq', 'fun': lambda x: np.sqrt((x[0] - 15)**2 + (x[1] - 30)**2) - 6},
        {'type': 'ineq', 'fun': lambda x: np.sqrt((x[0] - 20)**2 + (x[1] - 25)**2) - 7},
        {'type': 'ineq', 'fun': lambda x: np.sqrt((x[0] - 25)**2 + (x[1] - 15)**2) - 8})    # 约束条件('eq'表示等于0，'ineq'表示大于等于0)
bounds = ((10, 25), (15, 30))    # 变量的上下界,由等式约束条件产生的约束范围要补充


# 蒙特卡罗法估计最优解
N = 1000    # 蒙特卡罗模拟次数
n = len(bounds)    # 变量个数
x_min = np.zeros(n)    # 最优解
obj_fun_min = np.inf    # 目标函数的最小值
for i in range(N):
    x = np.zeros(n)
    # 随机生成变量
    while True:
        for j in range(2):    # 循环几次决定于等式约束条件的秩，示例情况下决策变量只有两个自由度
            if(bounds[j][0] == None and bounds[j][1] == None):
                x[j] = np.random.exponential(1.0) - np.random.exponential(1.0)
            elif(bounds[j][0] == None):
                x[j] = bounds[j][1] - np.random.exponential(1.0)
            elif(bounds[j][1] == None):
                x[j] = np.random.exponential(1.0) + bounds[j][0]
            else:
                x[j] = np.random.uniform(bounds[j][0], bounds[j][1])
        # 根据等式约束条件补足变量

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
""" 
minimize函数method参数：
一、无约束问题
    （1）不需要梯度信息
        1. 'Nelder-Mead'（也称为单纯形法）:一种启发式算法。
        2. 'Powell':一种共轭方向方法。
    （2）需要目标函数的一阶导数
        1. 'CG'（共轭梯度法）。
        2. 'BFGS':BFGS方法是一种拟牛顿方法。
    （3）需要目标函数的二阶导数
        1. 'trust-exact':信赖域精确线性编程。
        2. 'Newton-CG':牛顿共轭梯度法。
        3. 'dogleg':对偶单纯形法。
        4. 'trust-ncg':信赖域牛顿法。
        5. 'trust-krylov': 信赖域Krylov方法。
二、有边界约束问题
    （1）需要目标函数的一阶导数
        1. 'TNC':Truncated Newton（截断牛顿）算法。
        2. 'L-BFGS-B':限制内存的BFGS方法。
三、有边界和等式/不等式约束的问题
    （1）不需要梯度信息
        1. 'COBYLA':约束优化方法。
    （2）需要目标函数的一阶导数
        1. 'SLSQP':序列二次规划方法。
        2. 'trust-constr':信赖域约束优化方法。
"""
res = minimize(obj_fun, x_min, method = 'SLSQP', bounds = bounds, constraints = cons)
if res.success:
    print('数学求解的最优解:', res.x)  # 最优解
    print('数学求解的目标函数的最小值:', res.fun)  # 目标函数的最小值
else:
    print('数学求解失败！')