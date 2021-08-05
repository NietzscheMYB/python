from numpy import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()


def classify0(inX, dataSet, labels, k):
    '''
    分类器
    :param inX: 用于分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签向量，标签向量元素数目和矩阵dataSet的行数相同
    :param k: 用于选择最近邻居的数目
    :return: 排序首位的label


    对未知类别属性的数据集中的每个点依次执行以下操作：
    1、计算已知类别数据集中的点与当前点之间的距离
    2、按照距离递增次序排序
    3、选取与当前点距离最小的 k 个点
    4、确定前 k 个点所在类别的出现频率
    5、返回前 k 个点出现频率最高的类别作为当前点的预测分类
    '''

    # ndarray.shape 数组维度的元组，ndarray.shape[0]表示数组行数，ndarray.shape[1]表示列数
    dataSetSize = dataSet.shape[0]
    # print(dataSetSize)

    # 将输入的 inX（1*2） 进行扩展，扩展为 4*2 矩阵，使其与训练样本集中的数据（4*2）矩阵作减法
    # 前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一行里有四个一样的），然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个就是坐标相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print(diffMat)

    # 上一行得到了坐标相减，然后这里要(x1-x2)^2，要求乘方
    # 将 差值矩阵 的每一项乘方
    sqDiffMat = diffMat ** 2

    # axis=1 计算出每一行的和
    # axis＝0表示按列相加，axis＝1表示按照行的方向相加，这样得到了(x1-x2)^2+(y1-y2)^2
    sqDistances = sqDiffMat.sum(axis=1)

    # 开根号，这个之后才是距离
    distances = sqDistances ** 0.5

    # argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
    sortedDistIndicies = distances.argsort()  # 返回的list

    # classCount 字典用于类别统计
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    soredClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return soredClassCount[0][0]


ans = classify0([0, 0], group, labels, 3)


"""
在约会网站上使用 kNN
1.收集数据： 提供文本文件
2.准备数据： 使用 Python 解析文本文件
3.分析数据： 使用 Matplotlib 画二维扩散图
4.训练算法： 此步骤不适合 k-近邻算法
5.测试算法：
    测试样本与非测试样本的区别在于：
        测试样本是已经完成分类的数据，如果预测分类与实际类别不用，则标记为一个错误
6.使用算法： 产生简单的命令行程序，然后可以输入一些特征数据以判断对方是否为自己喜欢的类型
"""
def file2matrix(filename):
    '''
        将文本记录转换NumPy的代码
    :param filename: 文件名
    :return: 转换后的矩阵
    '''
    fr = open(filename)

    # 将文件内容按行读取为一个list
    arrayOLines = fr.readlines()

    # 获取list的长度，即文件内容的行数
    numberOfLines = len(arrayOLines)

    # 生成numberOfLines*3 的零矩阵
    returnMat = np.zeros((numberOfLines, 3))

    # 分类标签 向量
    classLabelVector = []

    #
    index = 0

    # 遍历读入文件的每一行
    for line in arrayOLines:

        #截取调所有的回车符
        line = line.strip()

        # 将 line 以空格符进行分割，分开后赋给listFromLine
        listFromLine = line.split('\t')

        # 将index 行所有元素替换为 listFromLine 中的 [0:3] ,其中[0:3]是左闭右开
        returnMat[index,:] = listFromLine[0:3]

        # 分类标签向量 list 中添加 listFromLine 中的最后一项
        # labels = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
        # classLabelVector.append(labels[listFromLine[-1]])
        classLabelVector.append(int(listFromLine[-1]))

        index +=1

    return returnMat,classLabelVector



datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

def get_figure(datingDataMat,datingLabels):
    '''
    直接浏览文本文件方法非常不友好，一般会采用图形化方式展示数据
    :param datingDataMat: 
    :param datingLabels: 
    :return: 
    '''

    fig = plt.figure()
    # “111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”。
    ax =fig.add_subplot(111)

    # 使用 datingDataMat 矩阵的第二、第三列数据
    # 分别表示特征值“玩视频游戏所消耗时间百分比”和“每周所消费的冰淇淋公升数”
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])

    # 利用变量 datingLabels 存储的类标签属性，在散点图上绘制色彩不等，尺寸不同的点
    # scatter plot 散点图
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
               15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()


get_figure(datingDataMat,datingLabels)


"""
    数据的归一化处理
"""
def autoNorm(dataSet):
    """
    方程中数字差值最大的属性对计算结果的影响最大，在处理这种不同范围的特征值时，采用数值归一化的方法
    :param dataSet: 输入的数据集
    :return: 归一化后的数据集合
    """

    # dataSet.min(0) 中的参数 0 使得函数可以从列中选取最小值，而不是选当前行的最小值
    # minVals 储存每列中的最小值
    minVals = dataSet.min(0)

    # maxVals 储存每行中的最小值
    maxVals = dataSet.max(0)

    # 求得差值
    ranges = maxVals - minVals

    #
    normDataSet = np.zeros(np.shape(dataSet))

    # 将数据集 dataSet 的行数放入m
    m = dataSet.shape[0]

    # 归一化
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals



# normDataSet,ranges,minVals = autoNorm(datingDataMat)
# print(normDataSet)
# print(ranges)
# print(minVals)


def datingClassTest():

    # 选择10% 的数据作为测试数据，90%的数据作为训练数据
    hoRatio =0.10

    #将输入的文件转换为 矩阵形式
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

    #数据归一化处理
    normDataSet,ranges,minVals = autoNorm(datingDataMat)

    #计算测试向量的数量
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)

    #错误数量统计
    errorCount = 0.0
    # 遍历 测试向量
    for i in range(numTestVecs):

        # #取数据集 的后10%为测试数据，错误率为5%

        # 调用 classify0() 函数
        # 以归一化后的的数据集 normDataSet 的第 i 行数据作为测试数据，
        # 以 numTestVecs:m 行数据作为训练数据，
        # datingLabels[numTestVecs:m] 作为标签向量，
        # 选择最近的 3 个邻居
        classifierResult = classify0(normDataSet[i, :], normDataSet[numTestVecs:m, :],datingLabels[numTestVecs:m], 3)

        #打印 预测结果 与 实际结果
        print("the classifier came back with: %d," "the real answer is: %d " % (classifierResult, datingLabels[i]))


        # # -----------------------------------------------------------------------
        # # 取 数据集 的后 10% 作为测试数据，错误率为 6%
        # classifierResult = classify0(normDataSet[m-numTestVecs+i, :], normDataSet[:m-numTestVecs, :],
        #                              datingLabels[:m-numTestVecs], 3)
        #
        # print("the classifier came back with: %d,"
        #       "the real answer is: %d " % (classifierResult, datingLabels[m-numTestVecs+i]))
        #
        # if classifierResult != datingLabels[m-numTestVecs+i]:
        #     errorCount += 1.0
        # # -----------------------------------------------------------------------


        # 当预测失败时，错误数量 errorCount += 1
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is : %f" % (errorCount/float(numTestVecs)))

# datingClassTest()

def classifyPerson():
    # 预测结果 list
    resultList = ['not at all', 'in small doses', 'in large doess']

    # 获取用户输入
    percentTats = float(input('percentage of time spent playing video games?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    ffMile = float(input('frequent flier miles earned per year?'))

    # 归一化数据集
    normDataSet, ranges, minVals = autoNorm(datingDataMat)

    # 将用户输入转化为一个 Matrix
    inArr = np.array([ffMile, percentTats, iceCream])

    # 调用 classify0() ，将用户输入矩阵归一化后进行运算
    classifierResult = classify0((inArr - minVals)/ranges, normDataSet, datingLabels, 3)

    # 打印预测结果
    print('You will probably like this person:', resultList[classifierResult-1])

classifyPerson()