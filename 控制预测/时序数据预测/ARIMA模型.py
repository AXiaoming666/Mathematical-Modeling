from pmdarima import auto_arima

class ARIMA_Model:
    def __init__(self, time_series_data):
        self.time_series_data = time_series_data

    # 使用auto_arima函数找到最佳参数
    def find_best_model(self):
        best_model = auto_arima(self.time_series_data,
                                start_p=1, max_p=3, 
                                start_q=1, max_q=3, 
                                start_P=1, max_P=2, 
                                start_Q=1, max_Q=2,
                                start_d=1, max_d=1,
                                start_D=1, max_D=2,
                                # 这是季节性周期，如果有的话
                                m=12,
                                seasonal=False, 
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

        return best_model

    # 使用最佳参数进行预测
    def predict(self, best_model, forecast_step):
        forecast_data = best_model.predict(n_periods=forecast_step)
        return forecast_data
    
if __name__ == '__main__':
    time_series_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    arima_model = ARIMA_Model(time_series_data)
    best_model = arima_model.find_best_model()
    forecast_data = arima_model.predict(best_model, 5)
    print(forecast_data)