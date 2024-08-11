import random
import math

class SimulatedAnnealing:
    # 初始化
    def __init__(self, T, cooling_rate, max_iter, obj_func, bounds, constraints,disturbance):
        self.T = T    # 初始温度
        self.cooling_rate = cooling_rate    # 退火速率
        self.max_iter = max_iter    # 最大迭代次数
        self.obj_func = obj_func    # 目标函数
        self.bounds = bounds    # 变量取值范围
        self.constraints = constraints    # 约束条件
        self.disturbance = disturbance    # 扰动幅度

    # 迭代主体
    def run(self, initial_solution):
        current_solution = initial_solution    # 将初始解赋给当前解
        best_solution = current_solution    # 记录最优解
        best_obj_value = self.obj_func(best_solution)    # 记录最优解的目标函数值
        # 迭代
        for i in range(self.max_iter):
            new_solution = self.generate_new_solution(current_solution)    # 生成新解
            new_obj_value = self.obj_func(new_solution)    # 计算新解的目标函数值
            delta_obj = new_obj_value - best_obj_value    # 计算新解与最优解的目标函数差值
            # 接受或拒绝新解
            if delta_obj < 0 or (delta_obj >= 0 and random.random() < math.exp(-delta_obj / self.T)):
                # 接受新解
                current_solution = new_solution
                # 更新最优解
                if new_obj_value < best_obj_value:
                    best_solution = new_solution
                    best_obj_value = new_obj_value
            # 更新温度
            self.T *= self.cooling_rate
        return best_solution

    # 生成新解
    def generate_new_solution(self, current_solution):
        new_solution = [0] * len(current_solution)    # 初始化新解
        # 随机扰动
        for i in range(len(current_solution)):
            new_solution[i] = current_solution[i] + random.uniform(-self.disturbance, self.disturbance)
        # 判断新解是否满足约束条件
        if not(all(self.constraints[j]['fun'](new_solution) >= 0 for j in range(len(self.constraints))) and all(new_solution[j] >= self.bounds[j][0] and new_solution[j] <= self.bounds[j][1] for j in range(len(self.bounds)))):
            return self.generate_new_solution(current_solution)    # 若新解不满足约束条件，则重新生成
        else:
            return new_solution


if __name__ == '__main__':
    obj_func = lambda x: (x[0] - 5) ** 2
    bounds = [(0, 10)]
    constraints = [{'type': 'ineq', 'fun': lambda x: x[0] - 2},]
    sa = SimulatedAnnealing(T=100, cooling_rate=0.99, max_iter=100000, obj_func=obj_func, bounds=bounds, constraints=constraints, disturbance=0.1)
    initial_solution = [4]
    best_solution = sa.run(initial_solution)
    print(best_solution)