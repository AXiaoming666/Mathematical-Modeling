import numpy as np
import inspect

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, fitness_function, bounds):
        self.population_size = population_size    # 种群大小
        self.mutation_rate = mutation_rate    # 变异概率
        self.fitness_function = fitness_function    # 适应度函数
        self.bounds = bounds    # 变量取值范围
        self.population = []    # 种群
        self.best_individual = None    # 最佳个体
        self.best_fitness = float('-inf')    # 最佳适应度

    # 生成初始种群
    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = []
            for j in range(len(inspect.signature(self.fitness_function).parameters)):    # 获取适应度函数参数个数
                individual.append(np.random.uniform(*self.bounds[j]))    # 个体要符合变量取值范围
            self.population.append(individual)    # 添加到种群中
    
    # 获取当前种群中适应度最高的个体
    def evaluate_fitness(self):
        for individual in self.population:
            fitness = self.fitness_function(*individual)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual

    # 选择亲本
    def select_parents(self):
        fitness_values = [self.fitness_function(*individual) for individual in self.population]    # 计算每个个体的适应度
        total_fitness = sum(fitness_values)    # 计算总适应度
        probabilities = [fitness / total_fitness for fitness in fitness_values]    # 归一化作为选择概率
        # 在种群中不重复地选择两个个体
        parents = [self.population[i] for i in np.random.choice(range(len(self.population)), size=2, replace=False, p=probabilities)]
        return parents
    
    # 遗传（遗传逻辑可以改进）
    def heredity(self, parent1, parent2):
        child = []
        # 子代的基因随机继承父代
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    # 交配（交配逻辑可以改进）
    def copulate(self):
        # 自由交配
        for j in range(self.population_size):
            parents = self.select_parents()
            child = self.heredity(*parents)
            child = self.mutate(child)
            self.population.append(child)

    # 变异
    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                individual[i] = np.random.uniform(*self.bounds[i])
        return individual
    
    # 自然选择（选择逻辑可以改进）
    def natural_selection(self):
        for i in range(self.population_size):
            self.population.remove(min(self.population, key=lambda x: self.fitness_function(*x)))

    # 算法主体
    def run(self, generations):
        self.generate_initial_population()    # 生成初始种群
        for i in range(generations):
            # 若在本文件内调试，则计算并打印每一代的最佳个体
            if __name__ == '__main__':
                self.evaluate_fitness()
                print(f"Generation {i+1}: Best fitness = {self.best_fitness}, Best individual = {self.best_individual}")
            self.copulate()    # 交配
            self.natural_selection()    # 自然选择
        self.evaluate_fitness()    # 计算最后一代的最佳个体
        return self.best_individual, self.best_fitness
    

if __name__ == '__main__':
    fitness_function = lambda x, y: - (x - 1) ** 2 - (y + 2) ** 2
    bounds = [(-100, 100), (-100, 100)]
    ga = GeneticAlgorithm(population_size=100, mutation_rate=0.2, fitness_function=fitness_function, bounds=bounds)
    best_individual, best_fitness = ga.run(generations=1000)
    print(f"Best individual = {best_individual}, Best fitness = {best_fitness}")