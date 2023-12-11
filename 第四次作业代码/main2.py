from functions import *


def main1():
    # 删除'Uniformity of Cell Shape'和'Bland Chromatin'
    new_X = X.drop(['Uniformity of Cell Shape', 'Bland Chromatin'], axis=1)
    new_X_train, new_X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=42)
    clf = tree.DecisionTreeClassifier(random_state=75)
    clf = clf.fit(new_X_train, y_train)
    y_pred = clf.predict(new_X_test)
    y_true = y_test
    report = classification_report(y_true, y_pred)
    print(report)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.4f}".format(accuracy))  # Accuracy: 0.9429
    draw_tree2(clf, '删正3倒3')


def draw_tree2(clf, name):
    feature_name = ['Clump Thickness', 'Uniformity of Cell Size',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                    'Normal Nucleoli', 'Mitoses']
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_name,
                                    class_names=["benign", "malignant"], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    save_dir = 'trees'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'tree_{name}')
    graph.render(filename=file_path, format='png')
    print(f"图片 tree_{name}.png 已保存至/trees")


if __name__ == "__main__":
    main1()
