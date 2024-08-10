class kruskal:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.parent = {}
        self.rank = {}
        self.mst = []

    def make_set(self, v):
        self.parent[v] = v
        self.rank[v] = 0

    def find(self, v):
        if self.parent[v]!= v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, v1, v2):
        root1 = self.find(v1)
        root2 = self.find(v2)
        if root1!= root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1

    def kruskal(self):
        for v in self.vertices:
            self.make_set(v)
        self.edges.sort(key=lambda x: x[0])
        for edge in self.edges:
            weight, v1, v2 = edge
            if self.find(v1)!= self.find(v2):
                self.mst.append(edge)
                self.union(v1, v2)
        return self.mst
    
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
    kruskal = kruskal(vertices, edges)
    print(kruskal.kruskal())