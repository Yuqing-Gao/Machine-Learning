import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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


def draw_confusion_matrix(cm, model):
    """
    :param cm: 混淆矩阵
    :param model: 模型
    :return: 该模型混淆矩阵图
    """
    model_name = type(model).__name__
    plt.figure(figsize=(8, 6), dpi=100)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


def val_model(mdl):
    """
    :param mdl: 模型
    :return: 在测试集上的混淆矩阵和分类报告
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 拟合数据
    mdl.fit(X_train, y_train)
    # 预测测试集数据
    y_pred = mdl.predict(X_test)
    # 计算分类准确率
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    draw_confusion_matrix(cm, mdl)
    # 绘制分类报告
    # cr = classification_report(y_test, y_pred)
    # print(cr)


def cross_val_model(mdl, n_splits=5):
    """
    :param mdl: 模型
    :param n_splits: 交叉验证折数
    :return: n折测试集平均的混淆矩阵和分类报告
    """
    # 初始化交叉验证模型
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 记录每个fold的结果
    acc_list = []
    cm_list = []
    cr_list = []

    # 迭代每个fold
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 拟合数据
        mdl.fit(X_train, y_train)
        # 预测测试集数据
        y_pred = mdl.predict(X_test)
        # 计算分类准确率
        accuracy = accuracy_score(y_test, y_pred)
        acc_list.append(accuracy)
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        cm_list.append(cm)
        # 计算分类报告
        # cr = classification_report(y_test, y_pred)
        # cr_list.append(cr)

    # 输出平均结果
    print('Average accuracy:', sum(acc_list) / n_splits)
    # draw_confusion_matrix(sum(cm_list), mdl)
    # print('Average classification report:')
    # print(' '.join(cr_list))


def cross_val_model2(mdl, seed, n_splits=5,):
    """
    :param seed: 随机种子数
    :param mdl: 模型
    :param n_splits: 交叉验证折数
    :return: n折测试集平均的混淆矩阵和分类报告
    """
    # 初始化交叉验证模型
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 记录每个fold的结果
    acc_list = []

    # 迭代每个fold
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 拟合数据
        mdl.fit(X_train, y_train)
        # 预测测试集数据
        y_pred = mdl.predict(X_test)
        # 计算分类准确率
        accuracy = accuracy_score(y_test, y_pred)
        acc_list.append(accuracy)

    # 输出平均结果
    return sum(acc_list) / n_splits

