from mlxtend.frequent_patterns import fpgrowth
import pandas as pd

# 假设我们有以下交易数据
data = [['bread', 'milk', 'butter'],
        ['bread', 'milk', 'eggs'],
        ['bread', 'milk'],
        ['milk', 'butter', 'eggs'],
        ['bread', 'butter', 'eggs'],
        ['bread', 'milk', 'butter', 'eggs']]

# 直接使用Pandas DataFrame，并将每个交易转换为布尔值
dummies = pd.DataFrame([[item in transaction for item in data[0]] for transaction in data], columns=data[0])

# 应用FP-Growth算法
frequent_itemsets = fpgrowth(dummies, min_support=0.5, use_colnames=True)

print(frequent_itemsets)