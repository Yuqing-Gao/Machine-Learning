import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("breast-cancer-wisconsin.csv", na_values=["?"])
df = df.fillna(round(df.mean(), 0))  # 均值填充缺失值
data = df.iloc[:, 1:]  # 删除第一列id号
data.columns = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data.to_csv("processed_data.csv", index=False)

# 划分特征和目标变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

