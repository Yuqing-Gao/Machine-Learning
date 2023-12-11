from functions import *


def main():
    # find_best_param()  # 谨慎运行，需要很长时间
    # 模型最佳参数: {'criterion': 'gini',
    # 'max_depth': 5, 'min_samples_leaf': 7, 'min_samples_split': 5,
    # 'random_state': 75, 'splitter': 'random'}

    # clf = generate_original_tree()
    # draw_tree(clf, '不剪枝的树')
    # 获取特征重要性
    # print_importance(clf, X_train)

    # clf1 = best_param_tree()
    # draw_tree(clf1, '最佳参数树')
    # print_importance(clf1, X_train)

    # 查看ccp路径和不纯度
    # pruning_path = clf.cost_complexity_pruning_path(X_train, y_train)
    # print("ccp_alphas:", pruning_path['ccp_alphas'])
    # print("impurities:", pruning_path['impurities'])

    # find_best_alpha()

    # for i in [1, 4, 10]:
    #     clf2, accuracy = ccp_tree(0.01 * i)
    #     draw_tree(clf2, f'alpha={0.01 * i}的树')

    clf3, accuracy = ccp_tree(0.04)


if __name__ == "__main__":
    main()
