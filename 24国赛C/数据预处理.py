import pandas as pd
import numpy as np


# 读入数据
excel1_sheet0 = pd.read_excel('.\\附件1.xlsx', sheet_name='乡村的现有耕地', header=0,  usecols='A,B,C')
excel1_sheet1 = pd.read_excel('.\\附件1.xlsx', sheet_name='乡村种植的农作物', header=0, usecols='A,B,C', skipfooter=4)    # 跳过说明部分
excel2_sheet0 = pd.read_excel('.\\附件2.xlsx', sheet_name='2023年的农作物种植情况', header=0, usecols='A,B,C,D,E,F')
excel2_sheet1 = pd.read_excel('.\\附件2.xlsx', sheet_name='2023年统计的相关数据', header=0, usecols='B,D,E,F,G,H', skipfooter=3)


# 修正excel2_sheet1与excel1_sheet0中‘普通大棚’的表示
excel2_sheet1.loc[excel2_sheet1['地块类型'] == '普通大棚 ', '地块类型'] = '普通大棚'
excel1_sheet0.loc[excel1_sheet0['地块类型'] == '普通大棚 ', '地块类型'] = '普通大棚'

# 补充excel2_sheet1中省略的农作物种植数据
for index, row in excel2_sheet1.loc[(excel2_sheet1['地块类型'] == '普通大棚 ') & (excel2_sheet1['种植季次'] == '第一季')].iterrows():
    row.at['种植地块'] = '智慧大棚'
    excel2_sheet1.loc[len(excel2_sheet1)] = row


# 销售单价取均值
for index, row in excel2_sheet1.iterrows():
    sales_price = row['销售单价/(元/斤)']
    if isinstance(sales_price, str) and '-' in sales_price:
        min_val, max_val = sales_price.split('-')
        mean_val = (float(min_val) + float(max_val)) / 2
        excel2_sheet1.at[index, '销售单价/(元/斤)'] = mean_val


# 作物的种植成本矩阵、销售单价矩阵和产量矩阵
planting_cost_matrix = np.zeros((59, 54))
planting_salesprice_matrix = np.zeros(59)
planting_yield_matrix = np.zeros((59, 54))
for index in range(59):
    # 检索数据
    if index < 15:
        loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == index + 1) & (excel2_sheet1['地块类型'] == '平旱地')]    # 选取对应作物的平旱地数据
        yield_ = loc['亩产量/斤']
        salesprice = loc['销售单价/(元/斤)']
        cost = loc['种植成本/(元/亩)']
        for i in range(6):
            planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']    # 单位种植成本乘地块面积
        for i in range(6):
            planting_salesprice_matrix[index] = salesprice
        for i in range(6):
            planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']    # 单位产量乘地块面积

        loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == index + 1) & (excel2_sheet1['地块类型'] == '梯田')]
        yield_ = loc['亩产量/斤']
        salesprice = loc['销售单价/(元/斤)']
        cost = loc['种植成本/(元/亩)']
        for i in range(6, 20):
            planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
        for i in range(6, 20):
            planting_salesprice_matrix[index] = salesprice
        for i in range(6, 20):
            planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']

        loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == index + 1) & (excel2_sheet1['地块类型'] == '山坡地')]
        yield_ = loc['亩产量/斤']
        salesprice = loc['销售单价/(元/斤)']
        cost = loc['种植成本/(元/亩)']
        for i in range(20, 26):
            planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
        for i in range(20, 26):
            planting_salesprice_matrix[index] = salesprice
        for i in range(20, 26):
            planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']

    elif index == 15:
        loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == index + 1) & (excel2_sheet1['地块类型'] == '水浇地') & (excel2_sheet1['种植季次'] == '单季')]
        yield_ = loc['亩产量/斤']
        salesprice = loc['销售单价/(元/斤)']
        cost = loc['种植成本/(元/亩)']
        for i in range(26, 34):
            planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
        for i in range(26, 34):
            planting_salesprice_matrix[index] = salesprice
        for i in range(26, 34):
            planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']

    elif index < 52:
        if index % 2 == 0:
            loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == 16 + (index - 16) / 2 + 1) & (excel2_sheet1['地块类型'] == '水浇地') & (excel2_sheet1['种植季次'] == '第一季')]
            yield_ = loc['亩产量/斤']
            salesprice = loc['销售单价/(元/斤)']
            cost = loc['种植成本/(元/亩)']
            for i in range(26, 34):
                planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
            for i in range(26, 34):
                planting_salesprice_matrix[index] = salesprice
            for i in range(26, 34):
                planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']

            loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == 16 + (index - 16) / 2 + 1) & (excel2_sheet1['地块类型'] == '普通大棚') & (excel2_sheet1['种植季次'] == '第一季')]
            yield_ = loc['亩产量/斤']
            salesprice = loc['销售单价/(元/斤)']
            cost = loc['种植成本/(元/亩)']
            for i in range(34, 50):
                planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
            for i in range(34, 50):
                planting_salesprice_matrix[index] = salesprice
            for i in range(34, 50):
                planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']

            loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == 16 + (index - 16) / 2 + 1) & (excel2_sheet1['地块类型'] == '普通大棚') & (excel2_sheet1['种植季次'] == '第一季')]
            yield_ = loc['亩产量/斤']
            salesprice = loc['销售单价/(元/斤)']
            cost = loc['种植成本/(元/亩)']
            for i in range(50, 54):
                planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
            for i in range(50, 54):
                planting_salesprice_matrix[index] = salesprice
            for i in range(50, 54):
                planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']

        else:
            loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == 16 + (index - 17) / 2 + 1) & (excel2_sheet1['地块类型'] == '智慧大棚') & (excel2_sheet1['种植季次'] == '第二季')]
            yield_ = loc['亩产量/斤']
            salesprice = loc['销售单价/(元/斤)']
            cost = loc['种植成本/(元/亩)']
            for i in range(50, 54):
                planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
            for i in range(50, 54):
                planting_salesprice_matrix[index] = salesprice
            for i in range(50, 54):
                planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']
    
    elif index < 55:
        loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == index - 18 + 1) & (excel2_sheet1['地块类型'] == '水浇地') & (excel2_sheet1['种植季次'] == '第二季')]
        yield_ = loc['亩产量/斤']
        salesprice = loc['销售单价/(元/斤)']
        cost = loc['种植成本/(元/亩)']
        for i in range(26, 34):
            planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
        for i in range(26, 34):
            planting_salesprice_matrix[index] = salesprice
        for i in range(26, 34):
            planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']

    else:
        loc = excel2_sheet1.loc[(excel2_sheet1['作物编号'] == index - 18 + 1) & (excel2_sheet1['地块类型'] == '普通大棚') & (excel2_sheet1['种植季次'] == '第二季')]
        yield_ = loc['亩产量/斤']
        salesprice = loc['销售单价/(元/斤)']
        cost = loc['种植成本/(元/亩)']
        for i in range(34, 50):
            planting_cost_matrix[index, i] = cost * excel1_sheet0.loc[i, '地块面积/亩']
        for i in range(34, 50):
            planting_salesprice_matrix[index] = salesprice
        for i in range(34, 50):
            planting_yield_matrix[index, i] = yield_ * excel1_sheet0.loc[i, '地块面积/亩']

# 各地块亩数
planting_area_matrix = np.zeros((54))
for index , row in excel1_sheet0.iterrows():
    planting_area_matrix[index] = row['地块面积/亩']

# 2023年种植情况矩阵
planting_2023_matrix = np.zeros((59, 54))
# 补充excel2_sheet0中合并单元格，但读入时显示nan的'种植地块'
for index, row in excel2_sheet0.iterrows():     # 遍历excel2_sheet0的所有行
    if pd.isna(row['种植地块']):    # 如果'种植地块'为nan
        for i in range(index - 1, 0, -1):    # 向上查找非nan的'种植地块'
            if not pd.isna(excel2_sheet0.iloc[i]['种植地块']):     # 找到非nan的'种植地块'
                excel2_sheet0.at[index, '种植地块'] = excel2_sheet0.at[i, '种植地块']    # 复制非nan的'种植地块'
                row['种植地块'] = excel2_sheet0.at[i, '种植地块']    # 复制非nan的'种植地块'
                break

# 填充2023年种植情况矩阵
    if row['种植地块'][0] == 'A':    # 平旱地
        planting_2023_matrix[row['作物编号'] - 1, int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[int(row['种植地块'][1:]) - 1]
    elif row['种植地块'][0] == 'B':    # 梯田
        planting_2023_matrix[row['作物编号'] - 1, 6 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[6 + int(row['种植地块'][1:]) - 1]
    elif row['种植地块'][0] == 'C':    # 山坡地
        planting_2023_matrix[row['作物编号'] - 1, 20 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[20 + int(row['种植地块'][1:]) - 1]
    elif row['种植地块'][0] == 'D':    # 水浇地
        if row['种植季次'] == '单季':
            planting_2023_matrix[row['作物编号'] - 1, 26 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[26 + int(row['种植地块'][1:]) - 1]
        elif row['种植季次'] == '第一季':
            planting_2023_matrix[(row['作物编号'] - 1 - 16) * 2 + 16, 26 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[26 + int(row['种植地块'][1:]) - 1]
        else:
            planting_2023_matrix[(row['作物编号'] - 1 + 18), 26 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[26 + int(row['种植地块'][1:]) - 1]
    elif row['种植地块'][0] == 'E':    # 普通大棚
        if row['种植季次'] == '第一季':
            planting_2023_matrix[(row['作物编号'] - 1 - 16) * 2 + 16, 34 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[34 + int(row['种植地块'][1:]) - 1]
        else:
            planting_2023_matrix[row['作物编号'] - 1 + 18, 34 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[34 + int(row['种植地块'][1:]) - 1]
    else:    # 智慧大棚
        if row['种植季次'] == '第一季':
            planting_2023_matrix[(row['作物编号'] - 1 - 16) * 2 + 16, 50 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[50 + int(row['种植地块'][1:]) - 1]
        else:
            planting_2023_matrix[(row['作物编号'] - 1 - 16) * 2 + 17, 50 + int(row['种植地块'][1:]) - 1] = row['种植面积/亩'] / planting_area_matrix[50 + int(row['种植地块'][1:]) - 1]

# 由于某些作物第一季或第二季销量缺失，建立回归模型分析第一季与第二季销量的关系，并填充planting_sales_matrix矩阵
# 建立线性回归模型
from sklearn.linear_model import LinearRegression

planting_sales_matrix = (planting_2023_matrix * planting_yield_matrix).sum(axis=1)
planting_sales1_matrix = []
planting_sales2_matrix = []
for i in range(16, 52, 2):
    if planting_sales_matrix[i] != 0 and planting_sales_matrix[i+1] != 0:
        planting_sales1_matrix.append(planting_sales_matrix[i])
        planting_sales2_matrix.append(planting_sales_matrix[i+1])

model1 = LinearRegression()
model2 = LinearRegression()
model1.fit(np.array(planting_sales1_matrix).reshape(-1, 1), np.array(planting_sales2_matrix))
model2.fit(np.array(planting_sales2_matrix).reshape(-1, 1), np.array(planting_sales1_matrix))


# 填充planting_sales_matrix矩阵
for i in range(16, 52, 2):
    if planting_sales_matrix[i] == 0:
        planting_sales_matrix[i] = model2.predict(np.array(planting_sales_matrix[i+1]).reshape(-1, 1))[0]
    elif planting_sales_matrix[i+1] == 0:
        planting_sales_matrix[i+1] = model1.predict(np.array(planting_sales_matrix[i]).reshape(-1, 1))[0]






# 保存矩阵
np.savetxt("planting_cost_matrix.csv", planting_cost_matrix, delimiter=",", fmt='%f')
np.savetxt("planting_yield_matrix.csv", planting_yield_matrix, delimiter=",", fmt='%f')
np.savetxt("planting_salesprice_matrix.csv", planting_salesprice_matrix, delimiter=",", fmt='%f')
np.savetxt("planting_2023_matrix.csv", planting_2023_matrix, delimiter=",", fmt='%f')
np.savetxt("planting_area_matrix.csv", planting_area_matrix, delimiter=",", fmt='%f')
np.savetxt("planting_sales_matrix.csv", planting_sales_matrix, delimiter=",", fmt='%f')