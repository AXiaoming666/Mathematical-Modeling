import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Ant_Colony_Optimization:
    def __init__(self, graph, alpha, beta, rho, Q, init_Q, Max_iter):
        self.graph = graph    # 图
        self.alpha = alpha    # 启发式因子
        self.beta = beta    # 信息素因子
        self.rho = rho    # 消散度因子
        self.Q = Q    # 信息素浓度
        self.init_Q = init_Q    # 初始信息素浓度
        self.Max_iter = Max_iter    # 最大迭代次数
        self.ant = {}    # 蚂蚁

    # 选择下一个节点
    def select_edge(self):
        # 取当前节点的邻边
        neighbors = self.graph.adj[self.ant['current_node']]
        accessed_nodes = []
        # 剔除访问过的节点
        for node in neighbors:
            if node not in self.ant['Visited_nodes']:
                accessed_nodes.append(node)
        # 判断是否有可访问的节点
        if len(accessed_nodes) == 0:
            return False
        # 计算备选节点的权重
        weights = [self.graph.edges[self.ant['current_node'], node]['pheromone'] ** self.alpha * (1 / self.graph.edges[self.ant['current_node'], node]['cost']) ** self.beta for node in accessed_nodes]
        # 将权重归一化，得到概率
        possibilities = [weights[i] / sum(weights) for i in range(len(weights))]
        # 按概率随机选择下一个节点
        next_node = np.random.choice([node for node in accessed_nodes], p=possibilities)
        # 更新信息
        self.ant['Visited_edges'].append((self.ant['current_node'], next_node))
        self.ant['Visited_nodes'].append(next_node)
        self.ant['Path_length'] += self.graph.edges[self.ant['current_node'], next_node]['cost']
        self.ant['current_node'] = next_node
        return True

    # 更新信息素
    def update_pheromone(self):
        for egde in self.graph.edges():
            self.graph.edges[egde[0], egde[1]]['pheromone'] *= (1 - self.rho)
        for edge in self.ant['Visited_edges']:
            self.graph.edges[edge[0], edge[1]]['pheromone'] += self.Q / self.ant['Path_length']

    # 初始化蚂蚁
    def init_ant(self):
        self.ant = {'Visited_nodes': [np.random.choice(list(self.graph.nodes()))], 'Visited_edges': [], 'Path_length': 0, 'current_node': None}
        self.ant['current_node'] = self.ant['Visited_nodes'][0]

    # 初始化背景信息素
    def init_pheromone(self):
        for edge in self.graph.edges():
            self.graph.edges[edge[0], edge[1]]['pheromone'] = self.init_Q

    # 算法主体
    def run(self):
        # 记录最短路径
        best_path_length = float('inf')
        best_path = []
        # 初始化蚁群和信息素
        self.init_ant()
        self.init_pheromone()
        # 迭代
        for i in range(self.Max_iter):
            # 初始化蚂蚁
            self.init_ant()
            # 遍历所有节点
            for j in range(len(self.graph.nodes()) - 1):
                if self.select_edge() == False:
                    break
            if (len(self.ant['Visited_nodes']) == len(self.graph.nodes())) and self.graph.has_edge(self.ant['Visited_nodes'][-1], self.ant['Visited_nodes'][0]):
                self.ant['Path_length'] += self.graph.edges[self.ant['Visited_nodes'][-1], self.ant['Visited_nodes'][0]]['cost']
                # 更新信息素
                self.update_pheromone()
                # 记录最短路径
                if self.ant['Path_length'] < best_path_length:
                    best_path_length = self.ant['Path_length']
                    best_path = self.ant['Visited_nodes']
            else:
                continue
        return self.graph, best_path_length, best_path
    
    # 打印带有路径长度的图
    def print_cost(self):
        # 生成图的布局
        pos = nx.spring_layout(self.graph)

        # 绘制节点和边
        nx.draw(self.graph, pos=pos, with_labels=True)

        # 生成标签
        labels = nx.get_edge_attributes(self.graph, 'cost')

        # 绘制标签
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

        # 显示图形
        plt.show()
    
    # 打印信息素图
    def print_pheromone(self):
        # 计算节点的布局
        pos = nx.spring_layout(self.graph)

        # 创建图形和轴
        fig, ax = plt.subplots()

        # 绘制图，边的宽度根据信息素浓度进行缩放
        edges, pheromones = zip(*nx.get_edge_attributes(self.graph, 'pheromone').items())
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=300, 
                edge_color=pheromones, width=2, edge_cmap=plt.cm.Blues, ax=ax, edge_vmin=min(pheromones), edge_vmax=max(pheromones))

        # 显示颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(pheromones), vmax=max(pheromones)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax)  # 指定与哪个轴相关联

        # 显示图形
        plt.show()


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edges_from([(0, 1, {'cost': 3}),
                      (0, 3, {'cost': 8}),
                      (0, 4, {'cost': 9}),
                      (1, 2, {'cost': 3}),
                      (1, 3, {'cost': 10}),
                      (1, 4, {'cost': 5}),
                      (2, 3, {'cost': 4}),
                      (2, 4, {'cost': 20}),
                      (3, 4, {'cost': 3})])
    alpha = 6
    beta = 1
    rho = 0.2
    Q = 1
    init_Q = 1000000
    num_ants = 100
    Max_iter = 1000

    ant_colony_optimization = Ant_Colony_Optimization(G, alpha, beta, rho, Q, init_Q, Max_iter)
    QG, best_path_length, best_path = ant_colony_optimization.run()
    print('最短路径长度:', best_path_length)
    print('最短路径:', best_path)
    ant_colony_optimization.print_pheromone()