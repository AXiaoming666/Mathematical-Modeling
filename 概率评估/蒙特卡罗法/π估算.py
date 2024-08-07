import matplotlib.pyplot as plt
import numpy as np


# 参数初始化，投放10000个点，圆半径为1，圆心坐标为（1，1）
N = 10000    # 投放点数
R = 1    # 圆半径
x_center, y_center = 1, 1    # 圆心坐标
n = 0    #    初始时还未投放点，有0个点在圆内


# 设置绘图窗口
plt.figure()
plt.title('Monte Carlo Pi Estimation')
plt.xlabel('x')
plt.ylabel('y')


# 保持绘图窗口，多次绘图
for i in range(N):    # 投放N个点
    # np.random.rand()返回一个[0,1)之间的随机数
    x = np.random.rand() * 2    # x坐标在[0,2)之间
    y = np.random.rand() * 2    # y坐标在[0,2)之间


    # 判断投放点是否在圆内
    if (x - x_center)**2 + (y - y_center)**2 <= R**2:    # 横纵坐标的平方和小于R的平方，则在圆内
        n += 1    # 点在圆内，n加1
        plt.plot(x, y, '.', color='blue')    # 圆内点用蓝色圆点表示
    else:    # 否则，在圆外
        plt.plot(x, y, '.', color='red')    # 圆外点用红色圆点表示


plt.axis('equal')    # 保持x、y轴比例相同,便于观察
plt.show()    # 显示绘图结果

# 计算圆周率π的估计值
pi_estimate = 4 * n / N    # 估计值
print('Estimated value of pi:', pi_estimate)    # 输出估计值