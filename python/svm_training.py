
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from utils_plot import plot_learning_curve


# 加载示例数据集
iris = np.load(r"TS_model\svm_training_z_fill.npz")

X = iris['X']
y = iris['y']  # 取前两个类别

# 划分训练集和测试集

print(X.shape, y.shape)
print(y.sum())
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# param_grid = {
#     'C': [0.2, 0.4, 0.7, 1, 1.5],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': [0.001, 0.003, 0.005, 0.01, 0.05, 0.1],
# }
# #
# grid_search = GridSearchCV(estimator=SVC(class_weight='balanced'),
#                            param_grid=param_grid, cv=5, n_jobs=8, scoring='f1', verbose=1)
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)

# 创建SVM分类器
# svm = SVC(kernel='rbf', C=1.5, random_state=30, gamma=0.002, probability=True, class_weight='balanced')
svm = SVC(kernel='rbf', C=0.3, random_state=30, gamma=0.01, probability=True)

# 训练模型
svm.fit(X_train, y_train)

# 在测试集上预测
y_pred = svm.predict(X_test)
# print(y_test)
# print(y_pred)

fig, ax = plt.subplots(figsize=(6, 4))
title = "Learning Curves (SVC)"

plt = plot_learning_curve(svm, title, X, y, ax=ax, cv=4, n_jobs=4)
plt.show()
# fig.savefig(r'TS_model\figure6_svm_learning_curves.png', dpi=300)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)


Y_pred = svm.predict(X_test)


FP = ((y_test == 0) & (Y_pred == 1)).sum()
FN = ((y_test == 1) & (Y_pred == 0)).sum()
TP = ((y_test == 1) & (Y_pred == 1)).sum()
TN = ((y_test == 0) & (Y_pred == 0)).sum()
print(f"FP: {FP}")
print(f"FN: {FN}")
print(f"TP: {TP}")
print(f"TN: {TN}")

precision = TP / (TP + FP)
recall = TP / (TP + FN)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"f1_score: {2*precision*recall/(precision+recall)}")
# if recall > 0.7 and precision > 0.7:
#     joblib.dump(svm, r'TS_model/svm_model.pkl')
#     joblib.dump(scaler, r'TS_model/scaler_model.pkl')

# 加载模型

