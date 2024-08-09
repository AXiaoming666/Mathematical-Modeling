import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize


# 修改matplotlib中字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义基本数据
# 目标函数(主意要消除量纲影响)
obj_fun1 = lambda x: 4*x[0] + 10*x[1]
obj_fun2 = lambda x: 12*x[0] + 9*x[1]
cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 7},)    # 约束条件
bounds = ((0, 5), (0, 6))    # 变量取值范围
span = (0, 1)    # 权重范围
step = 0.01    # 步长


# 定义线性加权法函数
for w in np.arange(span[0], span[1], step):
    obj_fun = lambda x: obj_fun1(x) * w + obj_fun2(x) * (1 - w)


    # 蒙特卡罗法估计最优解
    N = 1000    # 蒙特卡罗模拟次数
    n = len(bounds)    # 变量个数
    x_min = np.zeros(n)    # 最优解
    obj_fun_min = np.inf    # 目标函数的最小值
    for i in range(N):
        x = np.zeros(n)
        # 随机生成变量
        while True:
            for j in range(2):    # 循环几次决定于等式约束条件的秩，示例情况下决策变量只有一个自由度
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
    

    # 数学求解
    res = minimize(obj_fun, x_min, method = 'SLSQP', bounds = bounds, constraints = cons)
    if res.success:
        # 绘图
        # 权值w对目标函数的影响
        plt.subplot(1, 2, 1)
        plt.scatter(w, obj_fun1(res.x), c = 'r', label = 'obj_fun1')
        plt.scatter(w, obj_fun2(res.x), c = 'b', label = 'obj_fun2')
        plt.title('权值w对目标函数的影响')
        plt.xlabel('权值w')
        plt.ylabel('目标函数值')
        # 权值w对最优解的影响
        plt.subplot(1, 2, 2)
        plt.scatter(w, res.x[0], c = 'r', label = 'x1')
        plt.scatter(w, res.x[1], c = 'b', label = 'x2')
        plt.title('权值w对最优解的影响')
        plt.xlabel('权值w')
        plt.ylabel('最优解')


plt.show()