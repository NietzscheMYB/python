from sklearn.datasets import load_iris
import mglearn
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split
"""
线性模型
ŷ = w[0] * x[0] + w[1] * x[1] + … + w[p] * x[p] + b
这里x[0] 到 x[p] 表示单个数据点的特征（本例中特征个数为p+1），
 w 和 b 是学习模型的 参数，ŷ 是模型的预测结果。
"""
mglearn.plots.plot_linear_regression_wave()
plt.show()

from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
#train_test_split函数用于将矩阵随机划分为训练子集和测试子集，
# 并返回划分好的训练集测试集样本和训练集测试集标签。！！！
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
#“斜率”参数（w，也叫作权重或系数）被保存在 coef_ 属性中，
# 而偏移或截距（b）被保 存在 intercept_ 属性中：
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#训练集和测试集之间的性能差异是过拟合的明显标志，
# 因此我们应该试图找到一个可以控 制复杂度的模型

"""
Logistic回归，线性支持向量机SVM
"""
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import LinearSVC
X, y = mglearn.datasets.make_forge()
fig,axes=plt.subplots(1,2,figsize=(10,3))
for model,ax in zip([LinearSVC(),LogisticRegression()],axes):
    clf=model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=False,eps=0.5,ax=ax,alpha=.7)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()