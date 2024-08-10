import numpy as np
class Floyd:
    def __init__(self, datas):
        self.n = max(max(data[0], data[1]) for data in datas) + 1  # 节点数
        self.graph = [[(lambda x: 0 if x[0] == x[1] else np.inf)([i, j]) for j in range(self.n)] for i in range(self.n)]
        self.parents = [[i] * self.n for i in range(4)]  # 关键地方，i-->j 的父结点初始化都为i
        for u, v, c in datas:
            self.graph[u][v] = c	# 因为是有向图，边权只赋给self.graph[u][v]

    # 弗洛伊德算法
    def floyd(self):
        self.n = len(self.graph)
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.graph[i][k] + self.graph[k][j] < self.graph[i][j]:
                        self.graph[i][j] = self.graph[i][k] + self.graph[k][j]
                        self.parents[i][j] = self.parents[k][j]  # 更新父结点


    # 打印路径
    def print_path(self, i, j):
        if i != j:
            self.print_path(i, self.parents[i][j])
        print(j, end='-->')

    # 输出最短路径矩阵
    def print_costs_matrix(self):
        print('Costs:')
        for row in self.graph:
            for e in row:
                print('∞' if e == np.inf else e, end='\t')
            print()

    # 输出最短路径
    def print_shortest_path(self, i, j):
        print('\nPath:')
        print('Path({}-->{}): '.format(i, j), end='')
        self.print_path(i, j)
        print(' cost:', self.graph[i][j])

if __name__ == '__main__':
    # Data [u, v, cost]
    datas = [
        [0, 1, 2],
        [0, 2, 6],
        [0, 3, 4],
        [1, 2, 3],
        [2, 0, 7],
        [2, 3, 1],
        [3, 0, 5],
        [3, 2, 12],
    ]
    floyd = Floyd(datas)
    floyd.floyd()
    floyd.print_costs_matrix()
    floyd.print_shortest_path(0, 3)