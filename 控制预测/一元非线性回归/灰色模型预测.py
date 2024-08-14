import numpy as np

class GrayModel:
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        # 修正因子
        self.c = 0

    # 级比检验
    def data_check(self):
        # 计算级比序列
        ratios = [self.data[k - 1] / self.data[k] for k in range(1, self.n)]

        # 判断级比序列是否都落在可容覆盖内
        if all(ratios < np.exp(2 / (self.n + 1))) and all(ratios > np.exp(- 2 / (self.n + 1))):
            return True
        else:
            return False
        
    # 修正数据
    def data_correction(self):
        pass
    
    def fit(self):
        # 累加生成序列
        accumulation = np.cumsum(self.data)
        
        # 累加生成序列的一次紧邻均值
        mean_acc = (accumulation[:-1] + accumulation[1:]) / 2.0
        
        # 创建X矩阵和Y向量
        X = np.column_stack((-mean_acc, np.ones(self.n - 1)))
        Y = self.data[1:]
        
        # 求解GM(1,1)模型参数a和b
        a, b = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        self.a = a
        self.b = b

        # 计算拟合值
        self.predictions = self.predict(self.n)

    def predict(self, num):
        # 使用模型进行预测
        predict_accumulation = (self.data[0] - self.b / self.a) * np.exp(-self.a * np.arange(1, self.n + num))

        # 累加还原得到原始数据的预测值
        predictions = np.zeros(num)
        predictions[0] = self.data[0]
        for i in range(1, num):
            predictions[i] = predict_accumulation[i] - predict_accumulation[i - 1]
        return predictions
    
    def relative_error(self):
        # 计算预测值相对于真实值的相对误差
        max_error = 0
        for i in range(1, self.n):
            relative_error = abs((self.predictions[i] - self.data[i]) / self.data[i])
            if relative_error > max_error:
                max_error = relative_error
        return max_error
    
    def ratios_deviation(self):
        max_deviation = 0
        for i in range(1, self.n):
            ratios = self.data[i - 1] / self.data[i]
            ratios_deviation = np.abs(1 - ((1 - 0.5 * self.a) / (1 + 0.5 * self.a)) * ratios)
            if ratios_deviation > max_deviation:
                max_deviation = ratios_deviation
        return max_deviation


if __name__ == '__main__':
    # 随便举例子，用于演示灰色预测
    data = [34, 35, 36, 37, 38, 39]
    
    gm = GrayModel(data)
    print(gm.data_check())
    gm.fit()
    print(gm.relative_error())
    print(gm.ratios_deviation())
    print(gm.a, gm.b)
    print(gm.predict(9))