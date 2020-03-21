from sklearn.datasets import load_iris
import mglearn
import matplotlib.pyplot as plt
import numpy
# #load_iris返回的iris对象是一个Bunch对象，与字典非常相似，里面包含键和值
# iris_dataset=load_iris
#
# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# #print(iris_dataset)
#
# #target_names 要预测花的品种
# print("Target names:{}".format(iris_dataset["target_names"]))
# #feature_names键对应的值是一个字符串列表，对每一个特征进行说明
# print("Feature names:\n{}".format(iris_dataset["feature_names"]))
############################3
"""
forge数据集，模拟二分类数据集，有两个特征
"""
#生成数据集
X,y=mglearn.datasets.make_forge()
#数据集绘图
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["Class 0","Class 1"],loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
print("X.shape:{}".format(X.shape)) # X.shape 可以看出，这个数据集包含 26 个数据点和 2 个特征。

#######################################
"""
们用模拟的 wave 数据集来说明回归算法。
wave 数据集只有一个输入特征和一个连续的 目标变量（或响应），
后者是模型想要预测的对象。
"""
X,y=mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)##限制坐标范围
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
print("X:shape{}".format(X.shape))##求出多少个数据，每个数据有多少个特征
############################
"""
乳腺癌数据
。每个肿瘤都被标记为“良性”（benign，表示无害肿瘤）
或“恶 性”（ malignant，表示癌性肿瘤）
基于人体组织的测量数据来学习预测肿瘤是否 为恶性。
"""
from sklearn.datasets import  load_breast_cancer
##包含在 scikit-learn 中的数据集通常被保存为 Bunch 对象，
# 里面包含真实 数据以及一些数据集信息。关于 Bunch 对象，
# 你只需要知道它与字典很相 似，而且还有一个额外的好处，
# 就是你可以用点操作符来访问对象的值
# （比 如用 bunch.key 来代替 bunch['key']）。
cancer=load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
#输出cancer数据的个数，和特征
print("Shape of cancer data:\n{}".format(cancer.data.shape))
#print("Sample counts per class:\n{}".format(\
#{n: v for n, v in zip(cancer.target_names, \
#np.bincount(cancer.target))}))

"""
波士顿房价数据集。
与这个数据集相关的 任务是，
利用犯罪率、是否邻近查尔斯河、公路可达性等信息，
来预测 20 世纪 70 年代波 士顿地区房屋价格的中位数。
"""
from sklearn.datasets import load_boston
boston=load_boston()
print("boston shape\n{}".format(boston.data.shape))
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
###################3

"""
k近邻
k-NN 算法最简单的版本只考虑一个最近邻，
也就是与我们想要预测的数据点最近的训练 数据点。
预测结果就是这个训练数据点的已知输出
"""
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()


