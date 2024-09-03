import pandas as pd

# 加载工作簿
file_path1 = '工作簿.xlsx'
data = pd.read_excel(file_path1, header = 0)


data['销售额'] = data['销售单价'] * data['销售额']
