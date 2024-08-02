import numpy as np

# 输入基本信息
print('参评对象数目')
n = int(input())
print('指标数目')
m = int(input())
print('指标类型\t1：极大型 2：极小型 3：中间型 4：区间型 指标间以空格隔开:')
kind = input().split(' ')
# 输入指标矩阵
print('输入矩阵A(指标矩阵):')
A = np.zeros(shape=(n, m))
for i in range(n):
    A[i] = input().split(' ')
    A[i] = list(map(float, A[i]))
print('矩阵A为:\n{}'.format(A))


def mintomax(xmax, x):      # 极小转极大
    x = list(x)
    ans = [[xmax - k]for k in x]
    return np.array(ans)


def midtomax(xbest, x):     # 中间转极大
    x = list(x)
    h = [abs(k - xbest)for k in x]
    M = max(h)
    if M == 0:      # 防止出错
        M = 1
    ans = [[1 - k/M]for k in h]
    return np.array(ans)


def regtomax(xlow, xhigh, x):       # 区间转极大
    x = list(x)
    M = max(xlow - min(x), max(x) - xhigh)
    ans = []
    for k in range(len(x)):
        if x[k] < xlow:
            ans.append([1 - (xlow - x[k])/M])
        elif x[k] > xhigh:
            ans.append([1 - (x[k] - xhigh)/M])
        else:
            ans.append([1])
    return np.array(ans)


# 统一指标类型(正向化)
X = np.zeros(shape=(n, 1))
for i in range(m):
    if kind[i] == '1':
        v = np.array(A[:, i])
    elif kind[i] == '2':
        maxA = max(A[:, i])
        v = mintomax(maxA, A[:, i])
    elif kind[i] == '3':
        print("类型3：请输入最优值:")
        best = eval(input())
        v = midtomax(best, A[:, i])
    elif kind[i] == '4':
        print("类型4：请输入区间下限:")
        low = eval(input())
        print("类型4：请输入区间上限:")
        high = eval(input())
        v = regtomax(low, high, A[:, i])
    else:
        print("kind should be from 1 to 4!")
        exit(1)
    if i == 0:
        X = v.reshape(-1, 1)
    else:
        X = np.hstack([X, v.reshape(-1, 1)])
print("统一后指标矩阵X:\n{}".format(X))


# 标准化矩阵
X = X.astype('float')
for i in range(m):
    X[:, i] = X[:, i]/np.sqrt(sum(X[:, i]**2))
print("标准化后矩阵Z:\n{}".format(X))


# 计算距离
x_max = np.max(X, axis=0)
x_min = np.min(X, axis=0)
d_best = np.sqrt(np.sum(np.square((X - np.tile(x_max, reps=1))), axis=1))
d_worst = np.sqrt(np.sum(np.square((X - np.tile(x_min, reps=1))), axis=1))
print("每个指标最大值：", x_max)
print("每个指标最小值：", x_min)
print("d+:", d_best)
print("d-:", d_worst)


# 计算得分
s = d_worst/(d_best + d_worst)
score = 100*s/sum(s)
for i in range(n):
    print("第{num}个对象得分为{Score}".format(num=i+1, Score=score[i]))
# 排序
objection = [(i, score[i])for i in range(n)]


def takesecond(elem):
    return elem[1]


objection.sort(key=takesecond)
print("升序排序:", objection)
