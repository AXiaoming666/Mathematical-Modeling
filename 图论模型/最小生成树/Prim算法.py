from collections import defaultdict
from heapq import *

class Prim:
    def __init__(self, vertices, edges, start='D'):
        self.vertices = vertices
        self.edges = edges
        self.start = start
        self.adjacent_dict = defaultdict(list)  # 注意：defaultdict(list)必须以list做为变量
        for weight, v1, v2 in edges:
            self.adjacent_dict[v1].append((weight, v1, v2))
            self.adjacent_dict[v2].append((weight, v2, v1))
        self.minu_tree = []  # 存储最小生成树结果
        self.visited = [start]  # 存储访问过的顶点，注意指定起始点
        self.adjacent_vertexs_edges = self.adjacent_dict[start]
        heapify(self.adjacent_vertexs_edges)  # 转化为小顶堆，便于找到权重最小的边

    def prim_algorithm(self):
        while self.adjacent_vertexs_edges:
            weight, v1, v2 = heappop(self.adjacent_vertexs_edges)  # 权重最小的边，并同时从堆中删除。
            if v2 not in self.visited:
                self.visited.append(v2)  # 在used中有第一选定的点'A'，上面得到了距离A点最近的点'D',举例是5。将'd'追加到used中
                self.minu_tree.append((weight, v1, v2))
                # 再找与d相邻的点，如果没有在heap中，则应用heappush压入堆内，以加入排序行列
                for next_edge in self.adjacent_dict[v2]:  # 找到v2相邻的边
                    if next_edge[2] not in self.visited:  # 如果v2还未被访问过，就加入堆中
                        heappush(self.adjacent_vertexs_edges, next_edge)
        return self.minu_tree

if __name__ == '__main__':
    vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    edges = [(7, 'A', 'B'),
             (5, 'A', 'D'),
             (8, 'B', 'C'),
             (9, 'B', 'D'),
             (7, 'B', 'E'),
             (5, 'C', 'E'),
             (15, 'D', 'E'),
             (6, 'D', 'F'),
             (8, 'E', 'F'),
             (9, 'E', 'G'),
             (11, 'F', 'G'),
             ]
    prim_obj = Prim(vertices, edges, start='D')
    print(prim_obj.prim_algorithm())
