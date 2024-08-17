from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from data import X, y

# 训练模型并评估准确率
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 展示决策树
plt.figure(figsize=(6, 6))
plt.title("Decision Tree")
plt.xlabel("Feature Index")
plt.ylabel("Tree Depth")
plot_tree(tree_clf, filled=True)
plt.show()