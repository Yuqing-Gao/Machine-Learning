from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from preprocess import cross_val_model, val_model, cross_val_model2

# 线性判别
lda = LinearDiscriminantAnalysis()
# Logistic
lr = LogisticRegression()
# KNN
k = 9  # 设置 k 值为 9（最佳）
knn = KNeighborsClassifier(n_neighbors=k)
# 先验分布是高斯分布的朴素贝叶斯
nb = GaussianNB()
# 先验分布是多项式分布的朴素贝叶斯，不平滑
nb3 = MultinomialNB(alpha=0, force_alpha=True)
# 先验分布是多项式分布的朴素贝叶斯，Laplace 平滑
nb4 = MultinomialNB(alpha=1)

"""
直接测试：
Accuracy: 0.9571428571428572
Accuracy: 0.9571428571428572
Accuracy: 0.9714285714285714
Accuracy: 0.9642857142857143
"""
# val_model(lda)
# val_model(lr)
# val_model(knn)
# val_model(nb)

"""
五折交叉验证：
Average accuracy: 0.9570606372045221
Average accuracy: 0.9613669064748201
Average accuracy: 0.964213771839671
Average accuracy: 0.959917780061665
"""
# cross_val_model(lda)
# cross_val_model(lr)
# cross_val_model(knn)
# cross_val_model(nb)


# 拉普拉斯平滑
lst_accuracy = []

for i in range(1000):
    accuracy3 = cross_val_model2(nb3, i)
    accuracy4 = cross_val_model2(nb4, i)

    if accuracy3 != accuracy4:
        lst_accuracy.append(accuracy4-accuracy3)


print(max(lst_accuracy)-min(lst_accuracy))  # 0.007173689619732748

