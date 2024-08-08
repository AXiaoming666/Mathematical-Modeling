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
res = minimize(obj_fun, x_min, method = 'SLSQP', bounds = bounds, constraints = cons)
if res.success:
    print('数学求解的最优解:', res.x)  # 最优解
    print('数学求解的目标函数的最小值:', res.fun)  # 目标函数的最小值
else:
    print('数学求解失败！')