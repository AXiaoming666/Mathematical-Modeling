import numpy as np

class MCMC:
    def __init__(self, func, x0, step_size, n_iter):
        self.func = func
        self.x0 = x0
        self.step_size = step_size
        self.n_iter = n_iter

    def sample(self):
        x = self.x0
        for i in range(self.n_iter):
            x_new = x + np.random.normal(0, self.step_size, len(x))
            if self.func(x_new) < self.func(x):
                x = x_new
        return x

def f(x):
    return x[0]**2 + x[1]**2

x0 = np.array([1, 1])
step_size = 0.1
n_iter = 10000


mcmc = MCMC(f, x0, step_size, n_iter)
x_star = mcmc.sample()


print("The estimated optimal solution is:", x_star)