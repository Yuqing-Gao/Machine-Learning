from preprocess import *
import seaborn as sns

# 绘制直方图
sns.set(style="darkgrid")
X.hist(figsize=(12, 10), bins=20, xlabelsize=8, ylabelsize=8)
plt.savefig("hist1.png")

# 以良性恶性为分类绘制直方图
sns.set(style="darkgrid")
fig, axs = plt.subplots(3, 3, figsize=(12, 10))
axs = axs.ravel()

for i, col in enumerate(X.columns):
    axs[i].hist(X.loc[y == 2, col], bins=20, alpha=0.5, label='Benign')
    axs[i].hist(X.loc[y == 4, col], bins=20, alpha=0.5, label='Malignant')
    axs[i].set_title(col)
    axs[i].legend()

plt.tight_layout()
plt.savefig("hist2.png")

# 相关性矩阵和热力图
corr_matrix = X.corr()
plt.figure(figsize=(16, 14))
heatmap = sns.heatmap(corr_matrix, center=0, square=True, linewidths=.5, annot=True, fmt='.2f')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)
plt.savefig("heatmap.png")
