import pandas as pd
import numpy as np


# 读入数据
excel1_sheet0 = pd.read_excel('.\\附件1.xlsx', sheet_name='乡村的现有耕地', header=0,  usecols='A,B,C')
excel1_sheet1 = pd.read_excel('.\\附件1.xlsx', sheet_name='乡村种植的农作物', header=0, usecols='A,B,C', skipfooter=4)    # 跳过说明部分
excel2_sheet0 = pd.read_excel('.\\附件2.xlsx', sheet_name='2023年的农作物种植情况', header=0, usecols='A,B,C,D,E,F')
excel2_sheet1 = pd.read_excel('.\\附件2.xlsx', sheet_name='2023年统计的相关数据', header=0, usecols='B,D,E,F,G,H', skipfooter=3)

# 处理数据
excel2_sheet1['种植成本(元/斤)'] = excel2_sheet1['种植成本/(元/亩)'] / excel2_sheet1['亩产量/斤']
df_unique = excel2_sheet1.drop_duplicates(subset='作物编号')    # 按照'作物编号'列去重
df_sorted = df_unique.sort_values(by='作物编号')    # 按照'作物编号'列排序

# 检验2023年种植成本是否低于销售单价
for index, row in df_sorted.iterrows():
    sales_price = row['销售单价/(元/斤)']
    if isinstance(sales_price, str) and '-' in sales_price:
        min_val, max_val = sales_price.split('-')
        if row['种植成本(元/斤)'] >= float(min_val):
            print('2023年出现了低于种植成本的销售单价')
            break
else:
    print('2023年种植成本均高于销售单价')


# 计算2023年地块最多分为几份种植
max_part = 1
for index, row in excel2_sheet0.iterrows():
    if pd.isna(row['种植地块']):
        for i in range(index, 0, -1):
            if not pd.isna(excel2_sheet0.iloc[i]['种植地块']):
                row['种植地块'] = excel2_sheet0.iloc[i]['种植地块']
                break
    part = excel1_sheet0.loc[excel1_sheet0['地块名称'] == row['种植地块'], '地块面积/亩'].values[0] / row['种植面积/亩']
    if part > max_part:
        max_part = part
print('2023年地块最多分为{}份种植'.format(max_part))


# 计算2023年同种作物最多在几块地上种植
block = np.zeros([41, 2])
for index, row in excel2_sheet0.iterrows():
    if row['种植季次'] == '单季':
        block[row['作物编号'] - 1, 0] += 1
        block[row['作物编号'] - 1, 1] += 1
    elif row['种植季次'] == '第一季':
        block[row['作物编号'] - 1, 0] += 1
    else:
        block[row['作物编号'] - 1, 1] += 1
max_block = np.max(block)
print('2023年同种作物最多在{}块地上种植'.format(max_block))


# 判断同种作物同季出售价格是否相同
df_sorted = excel2_sheet1.sort_values(by=['作物编号', '种植季次'])
for index, row in df_sorted.iterrows():
    if index == 0:
        continue
    if row['作物编号'] == df_sorted.iloc[index-1]['作物编号'] and row['种植季次'] == df_sorted.iloc[index-1]['种植季次']:
        if row['销售单价/(元/斤)'] != df_sorted.iloc[index-1]['销售单价/(元/斤)']:
            print('2023年同种作物同季出售价格不同')
            break
else:
    print('2023年同种作物同季出售价格相同')