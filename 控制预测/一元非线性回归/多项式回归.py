import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree, x_data, y_data):
        self.degree = degree
        self.x_data = x_data
        self.y_data = y_data

    def fit(self):
        x = self.x_data
        for i in range(2, self.degree+1):
            x = np.column_stack((x, self.x_data**i))
        x = sm.add_constant(x)
        model = sm.OLS(self.y_data, x)
        result = model.fit()
        return result
    
    def print_graph(self,result):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(self.x_data, self.y_data, label='实际值')      # 散点图
        x = np.linspace(self.x_data.min(), self.x_data.max(), 100)
        plt.plot(x, sum(result.params[i]*x**i for i in range(self.degree+1)), color='red', label='预测值')  # 预测值
        plt.legend()        # 显示图例，即每条线对应label中的内容
        plt.show()      # 显示图形

    def run(self):
        result = self.fit()
        self.print_graph(result)


if __name__ == '__main__':
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 9, 16, 9, 4])
    degree = 3
    pr = PolynomialRegression(degree, x_data, y_data)
    pr.run()