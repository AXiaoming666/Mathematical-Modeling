import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# 修改matplotlib中字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2)


# 定义基本数据
# 目标函数(主意要消除量纲影响)
obj_fun1 = lambda x: 4*x[0] + 10*x[1]
obj_fun2 = lambda x: 12*x[0] + 9*x[1]
cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 7},)    # 约束条件
bounds = ((0, 5), (0, 6))    # 变量取值范围
span = (0, 1)    # 权重范围
step = 0.01    # 步长


# 蒙特卡罗法估计最优解
N = 1000    # 蒙特卡罗模拟次数
n = len(bounds)    # 变量个数
x1_min = np.zeros(n)    # 目标函数1最优解
x2_min = np.zeros(n)    # 目标函数2最优解
obj_fun1_min = np.inf    # 目标函数1的最小值
obj_fun2_min = np.inf    # 目标函数2的最小值
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
    obj_fun1_val = obj_fun1(x)    # 计算目标1函数的值
    if obj_fun1_val < obj_fun1_min:
        obj_fun_min = obj_fun1_val
        x1_min = x
    obj_fun2_val = obj_fun2(x)    # 计算目标2函数的值
    if obj_fun2_val < obj_fun2_min:
        obj_fun_min = obj_fun2_val
        x2_min = x


# 数学求解
res1 = minimize(obj_fun1, x1_min, method = 'SLSQP', bounds = bounds, constraints = cons)
res2 = minimize(obj_fun2, x2_min, method = 'SLSQP', bounds = bounds, constraints = cons)
if not res1.success or not res2.success:
    print('求解失败')
    exit()


# 理想点法求解
# 记录目标函数值和最优解
obj_fun1_values = []
obj_fun2_values = []
x1_values = []
x2_values = []
for w in np.arange(span[0], span[1], step):
    obj_fun = lambda x: (obj_fun1(x) - res1.fun)  ** 2 * w + (obj_fun2(x) - res2.fun) ** 2 * (1 - w)    # 定义加权平方和函数


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
        obj_fun1_values.append(obj_fun1(res.x))
        obj_fun2_values.append(obj_fun2(res.x))
        x1_values.append(res.x[0])
        x2_values.append(res.x[1])


# 绘图
ax1.plot(np.arange(span[0], span[1], step), obj_fun1_values, c = 'r')
ax1.plot(np.arange(span[0], span[1], step), obj_fun2_values, c = 'b')
ax1.set_title('权值w对目标函数的影响')
ax1.set_xlabel('权值w')
ax1.set_ylabel('目标函数值')
ax2.plot(np.arange(span[0], span[1], step), x1_values, c = 'r')
ax2.plot(np.arange(span[0], span[1], step), x2_values, c = 'b')
ax2.set_title('权值w对最优解的影响')
ax2.set_xlabel('权值w')
ax2.set_ylabel('最优解')


# 添加图例
lable1, = ax1.plot([], [], c = 'r', label = 'obj_fun1')
lable2, = ax1.plot([], [], c = 'b', label = 'obj_fun2')
lable3, = ax2.plot([], [], c = 'r', label = 'x1')
lable4, = ax2.plot([], [], c = 'b', label = 'x2')
ax1.legend(handles = [lable1, lable2])
ax2.legend(handles = [lable3, lable4])

# 调整子图间距并显示
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.2)
plt.show()