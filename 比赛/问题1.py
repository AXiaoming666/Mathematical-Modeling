import pandas as pd

# 加载数据
file_path1 = '.\\比赛\\附件1.xlsx'
data1 = pd.read_excel(file_path1, header = 0, usecols=['单品编码', '分类编码'], dtype={'单品编码':'str', '分类编码':'str'})
file_path2 = '.\\比赛\\附件2.xlsx'
data2 = pd.read_excel(file_path2, header = 0, nrows=40,usecols=['单品编码', '销售日期', '销量(千克)', '销售单价(元/千克)'], dtype={'单品编码':'str'})

# 处理数据
data2['销售额(元)'] = data2['销售单价(元/千克)'] * data2['销量(千克)']
data2.drop(['销售单价(元/千克)', '销量(千克)'], axis=1, inplace=True)

# 合并数据
data = pd.merge(data1, data2, on='单品编码', how='right')

# 整理数据
data_sorted = data.groupby(['单品编码', '销售日期'])['销售额(元)'].sum().reset_index()

# 将单品编码设置为列索引，销售日期设置为行索引
df = data_sorted.set_index(['销售日期', '单品编码']).unstack(level=-1)
print(df.columns)
print(df.index)