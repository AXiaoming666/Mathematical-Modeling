import numpy as np


# 在成功条件下的概率
n = 100000     # 试验次数
a = 0    # 不改变主意时能赢得汽车的次数
b = 0    # 改变主意时能赢得汽车的次数
c = 0    # 没能赢得汽车的次数
for i in range(n):
    x = np.random.randint(1, 4)    # 表示汽车出现在第x扇门后
    y = np.random.randint(1, 4)    # 表示自己选的门
    change = np.random.randint(0, 2)    # change = 1表示改变主意，change = 0表示不改变
    if x == y:    # 如果x == y，那么只有不改变主意时才能赢
        if change == 0:    # 不改变主意
            a += 1
        else:    # 改变主意
            c += 1
    else:    # 如果x != y，那么改变主意时才能赢
        if change == 1:    # 改变主意
            b += 1
        else:    # 不改变主意
            c += 1
print('不改变主意时能赢得汽车的概率:', a / n)
print('改变主意时能赢得汽车的概率:', b / n)
print('没能赢得汽车的概率:', c / n)