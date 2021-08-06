from math import log
import operator


# 划分数据集的大原则：将无序的数据变得更加有序（尽量使信息熵降低）
# 信息增益：在划分数据集前合信息熵发生的变化
# 获得 信息增益 最高的特征就是最好的选择
# 集合信息的 度量方式称为 香浓熵 或者 熵

def calcShannonEnt(dataSet):
    """
    计算给定数据集的 香农熵
    :param dataSet: 
    :return: 
    """
    # 计算输入数据集的 样本 总数
    numEntries = len(dataSet)

    # 创建一个用于统计 标签 出现 次数的dict
    labelCounts = {}

    # 遍历数据集中的 样本
    for featVec in dataSet:
        # 当前 样本 的标签 为 featVec[-1]
        currentLabel = featVec[-1]

        # 记录标签的出现次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1


    # 香浓熵 初始化
    shannonEnt = 0.0

    # 遍历字典 labelCounts 的key
    for key in labelCounts:

        #计算 标签 出现的频率 （概率）
        prob = float(labelCounts[key])/numEntries

        # 计算 香农熵
        shannonEnt -= prob * log(prob,2)

    return  shannonEnt

# #熵越高，则混合的数据越多
# print(calcShannonEnt(myDat))


def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flipperrs']
    return dataSet,labels


myDat,labels = createDataSet()

# print(calcShannonEnt(myDat))
# 熵越高，则混合数据越多，在数据集中添加更多的分类，观察熵是如何变化的




# 按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    """
    按照给定的特征划分数据集
    遍历数据集中的每一个元素，一旦发现符合要求的值，则将其添加到新创建的列表中
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征的项，是个下标
    :param value: 需要返回的特征的值
    :return: 划分后的数据集
    """
    retDataSet = []

    # 遍历数据集的每一个样本
    for featVec in dataSet:

        # 如果样本的 axis 项（指定特征）的值与输入的value相同
        if featVec[axis] == value:
            # 将 当前样本 featVec 中除了 axis 项（指定特征）之外的其余项保存到 reducedFeatVec list 中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])

            # 将reducedFeatVec 追加到 retDataSet中
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    选取特征，划分数据集，计算出最好的划分数据集的特征
    传入的dataSet满足：
        1.数据必须是一种由列表元素组成的list，而且所有的列表元素都要具有相同的数据长度
        2.数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
    :param dataSet: 
    :return: 
    """

    # 判定当前数据集包含多少个特征
    numFeatures = len(dataSet[0]) - 1

    #计算输入 dataSet的香农熵
    baseEntropy = calcShannonEnt(dataSet)

    # 初始化 最优 信息增益
    bestInfoGain = 0.0

    # 初始化 最优 特征
    bestFeature = -1

    # 遍历 样本 除了 类别标签 外的每一个 特征
    for i in range(numFeatures):

        # featureList 存放 当前 i 列的 特征值
        featureList = [example[i] for example in dataSet]

        # 去掉当前特征中的重复值
        uniqueVals = set(featureList)
        # print(uniqueVals)

        # 初始化新的香浓熵
        newEntropy =0.0

        #遍历 去掉重复值 后的特征值 list（以每个独一无二的属性去划分数据集）
        for value in uniqueVals:

            #划分数据集
            subDataSet = splitDataSet(dataSet,i,value)

            # 求得 划分后 数据集 占 原始数据集 的比重
            prob = len(subDataSet)/float(len(dataSet))

            # 将 比重 乘以 划分后 数据集 的香农熵 并 求和
            newEntropy += prob*calcShannonEnt(subDataSet)

        # 信息增益 等于 基本香浓熵 - 当前香浓熵
        infoGain = baseEntropy - newEntropy

        # 如果 信息增益 大于 最优信息增益
        if infoGain > bestInfoGain:

            # 最优信息增益 为 当前信息增益
            bestInfoGain = infoGain

            # 最优特征（列） 为当前列
            bestFeature = i

    return bestFeature


bestFeature = chooseBestFeatureToSplit(myDat)
print(bestFeature)


#返回出现次数最多的分类名称
def majorityCnt(classList):
    """
        如果数据集已经处理了所有 特征 ，但是 特征值 依然不是唯一的，此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方式。
    
    返回出现次数最多的分类名称
    :param classList: 类标签 list
    :return: 出现次数最多的 类标签
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#创建决策数
def createTree(dataSet,labels):
    """
    
    :param dataSet: 数据集
    :param labels: 标签列表，这块应该是每一列特征名称组成的list！
    :return: 
    """
    # classList 存储数据集所有 样本 的标签
    classList = [example[-1] for example in dataSet]

    # 当所有的标签完全相同的时候，直接返回该标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 当 使用完 所有特征，仍然不能将数据集划分成 仅包含 唯一类别 的分组
    if len(dataSet[0]) == 1:
        # 调用majority() 返回当前数据集，出现次数最多的标签
        return majorityCnt(classList)

    # 选择数据集中的 最优特征（列） （最优特征的序号）
    bestFeature = chooseBestFeatureToSplit(dataSet)

    # 最优特征（列）（序号）对应的 特征  （名称）
    bestFeatureLabel = labels[bestFeature]

    #存储树的所有信息
    myTree = {bestFeatureLabel:{}}

    #从labels list中删除最优特征
    del (labels[bestFeature])

    # 遍历dataSet 从中提取 最优特征(列）的所有 特征值
    featureValues = [example[bestFeature] for example in dataSet ]

    # 去掉重复值
    uniqueValues = set(featureValues)

    # 遍历独一无二的特征值
    for value in uniqueValues:

        # 将 剩下的label 存入 subLabels 中
        subLabels = labels[:]

        # 递归调用createTree() ，得到的返回值将被插入myTree中
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)

    return myTree

print(labels)
myTree = createTree(myDat,labels)
print(myTree)


# 使用决策树的分类函数
def classify(inputTree,featureLabels,testVec):
    """
    
    :param inputTree: 已经构造好的决策树 
    :param featureLabels: 特征标签的list
    :param testVec: 测试向量（各个特征组成的list）
    :return: 
    """
    # 根节点
    firstStr = list(inputTree.keys())[0]

    # 子节点
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            # 如果当前节点为 根节点
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featureLabels, testVec)
            # 如果当前节点为 叶子节点
            else:
                classLabel = secondDict[key]
    return classLabel

