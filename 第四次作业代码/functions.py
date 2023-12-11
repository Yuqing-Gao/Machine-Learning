import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from preprocess import *
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt


def generate_original_tree():
    clf = tree.DecisionTreeClassifier(random_state=75)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    report = classification_report(y_true, y_pred)
    print(report)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.4f}".format(accuracy))
    return clf


def best_param_tree():
    clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=7, min_samples_split=5,
                                      criterion="gini", random_state=75, splitter="random")
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    report = classification_report(y_true, y_pred)
    print(report)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.4f}".format(accuracy))
    return clf


def ccp_tree(ccp):
    clf = tree.DecisionTreeClassifier(random_state=75, ccp_alpha=ccp)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test

    # 打印分类报告
    report = classification_report(y_true, y_pred)
    print(report)

    # 打印accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print_importance(clf, X_train)
    return clf, accuracy


def draw_tree(clf, name):
    feature_name = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                    'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_name,
                                    class_names=["benign", "malignant"], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    save_dir = 'trees'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'tree_{name}')
    graph.render(filename=file_path, format='png')
    print(f"图片 tree_{name}.png 已保存至/trees")
    # graph.view()


def find_best_param():
    clf = tree.DecisionTreeClassifier(random_state=0)
    param_test = {
        'max_depth': range(1, 10, 1),  # 最大深度
        'min_samples_leaf': range(1, 10, 1),
        'random_state': range(0, 100, 5),
        'min_samples_split': range(5, 20, 3),
        'splitter': ('best', 'random'),
        'criterion': ('gini', 'entropy')
    }
    gsearch = GridSearchCV(estimator=clf,  # 对应模型
                           param_grid=param_test,  # 要找最优的参数
                           scoring=None,  # 准确度评估标准
                           n_jobs=-1,  # 并行数个数,-1:跟CPU核数一致
                           cv=5,  # 交叉验证 5折
                           verbose=0  # 输出训练过程
                           )
    gsearch.fit(X_train, y_train)
    print("模型最佳评分:", gsearch.best_score_)
    print("模型最佳参数:", gsearch.best_params_)


def find_best_alpha():
    alpha_values = [0.01 * i for i in range(32)]
    accuracies = []
    impurities = []

    for alpha in alpha_values:
        clf2, accuracy = ccp_tree(alpha)
        is_leaf = clf2.tree_.children_left == -1
        tree_impurity = (clf2.tree_.impurity[is_leaf] * clf2.tree_.n_node_samples[is_leaf] / len(y_train)).sum()
        accuracies.append(accuracy)
        impurities.append(tree_impurity)
        print(f"Impurity after setting alpha={alpha}:", tree_impurity)
        print("Accuracy: {:.4f}".format(accuracy))

    # 绘制准确率和不纯度的图表
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, accuracies, marker='o', label='Accuracy')
    plt.plot(alpha_values, impurities, marker='o', label='Impurity')
    plt.xlabel('Alpha')
    plt.ylabel('Value')
    plt.title('Accuracy and Impurity vs Alpha')
    plt.legend()
    plt.show()


def print_importance(clf, X):
    importance = clf.feature_importances_
    features = X.columns

    plt.figure(figsize=(10, 6))
    plt.bar(features, importance)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance of Decision Tree')
    plt.xticks(rotation=20, fontsize=6.5)
    plt.show()

