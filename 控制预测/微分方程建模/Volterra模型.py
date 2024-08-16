import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def volterra_model(X, t, alpha, beta, delta, gamma, c):
    P, V = X
    dPdt = alpha * P - beta * P * V - c * P
    dVdt = delta * P * V - gamma * V
    return [dPdt, dVdt]

P0 = 100
V0 = 20
t = np.linspace(0, 10, 100)
alpha = 1.0
beta = 0.1
delta = 0.1
gamma = 1.0
c = 0.05

# 求解微分方程组
X0 = [P0, V0]    # 初始状态
sol = odeint(volterra_model, X0, t, args = (alpha, beta, delta, gamma, c))
P, V = sol[:, 0], sol[:, 1]

# 绘图
plt.plot(t, P, label = 'Prey')
plt.plot(t, V, label = 'Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Volterra Model with Capture Intensity')
plt.legend()
plt.show()