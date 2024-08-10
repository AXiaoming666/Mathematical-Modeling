import random
import math

class SimulatedAnnealing:
    def __init__(self, T, cooling_rate, max_iter, obj_func, disturbance):
        self.T = T
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.obj_func = obj_func
        self.disturbance = disturbance

    def run(self, initial_solution):
        current_solution = initial_solution
        best_solution = current_solution
        best_obj_value = self.obj_func(best_solution)
        for i in range(self.max_iter):
            new_solution = self.generate_new_solution(current_solution)
            new_obj_value = self.obj_func(new_solution)
            delta_obj = new_obj_value - best_obj_value
            if delta_obj < 0 or (delta_obj >= 0 and random.random() < math.exp(-delta_obj / self.T)):
                current_solution = new_solution
                if new_obj_value < best_obj_value:
                    best_solution = new_solution
                    best_obj_value = new_obj_value
            self.T *= self.cooling_rate
        return best_solution


    def generate_new_solution(self, current_solution):
        new_solution = [0] * len(current_solution)
        for i in range(len(current_solution)):
                new_solution[i] = current_solution[i] + random.uniform(-self.disturbance, self.disturbance)
        return new_solution

if __name__ == '__main__':
    def obj_func(x):
        return x[0]**2 + 2 * x[0] + 1

    sa = SimulatedAnnealing(T=100, cooling_rate=0.99, max_iter=1000000, obj_func=obj_func, disturbance=0.1)
    initial_solution = [1]
    best_solution = sa.run(initial_solution)
    print(best_solution)