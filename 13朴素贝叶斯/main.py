import numpy as np
import random

#词表到向量的转换函数
def loadDataSet():
    """
    创建一些实验样本
    :return: 
    """
    # 进行词条切分后的文档集合
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签集合
    # 0 表示正常文档，1代表侮辱性文档
    # 标注信息用于训练程序一遍自动检测侮辱性留言
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

listOPosts,listClasses = loadDataSet()
# print(listOPosts)
# print(listClasses)

def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复词的list
    :param dataSet: 
    :return: 
    """
    # 创建一个空集
    vocabSet = set([])

    # 创建两个集合的并集
    for document in dataSet:
        # 将每篇文档返回的新词集合添加到该集合中
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)

    return list(vocabSet)

myVocabList = createVocabList(listOPosts)
# print(myVocabList)
#
index_stupid = myVocabList.index('stupid')
# print(index_stupid)

# 词集模型
def setOfWord2Vec(vocabList,inputSet):
    """
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 文档向量，向量的每个元素为 1 或 0，分布表示词汇表中的单词再输入文档中是否出现
    返回的是是对应词汇表的下标，该单词存在为1，不存在为0
    """

    # 创建一个其中所含元素都为 0 的向量，长度于词汇表相同
    returnVec = [0]*len(vocabList)

    # 遍历文档中的所有单词
    for word in inputSet:

        # 如果出现了词汇表中的单词
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s is not in my Vocabulary!", word)
    return returnVec

result_1 = setOfWord2Vec(myVocabList, listOPosts[0])
result_2 = setOfWord2Vec(myVocabList, listOPosts[3])
print(result_1)
print(result_2)
# print(listOPosts[0])
print(len(myVocabList))


def trainNB0(trainMatrix,trainCatagory):
    """
    
    :param trainMatrix: 训练文档矩阵，注意矩阵与列表的形式，这里是跟numpy！
    :param trainCatagory: 训练文档对应的标签
    :return: 
    """
    # 训练文档的总数
    numTrainDocs = len(trainMatrix)

    # 词汇表的长度（列数）
    numWords = len(trainMatrix[0])

    # 任意文档 属于 侮辱性 文档 的概率
    pAbusive = sum(trainCatagory)/float(numTrainDocs)

    # # 词汇表长度，以 0 填充的矩阵
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    #
    # # denom 分母项
    # p0Denom = 0.0
    # p1Denom = 0.0

    # 如果其中一个概率为0，那么最后乘积也为0
    # 为了降低这种影响，将所有词的出现数初始化为1，并将分母初始化为2
    p0Num = np.ones(numWords)  # 生成的是行向量
    p1Num = np.ones(numWords)   #生成的是行向量
    p0Denom =2.0
    p1Denom =2.0

    # 遍历训练文档集中的每一篇文档
    for i in range(numTrainDocs):
        # 如果该文档的分类为 侮辱性 文档
        if trainCatagory[i] == 1:
            # 文档矩阵相加，最后获得的p1Num 矩阵的每个元素为该词汇在所有文档中出现的总次数（一个行向量）
            p1Num += trainMatrix[i]
            # 矩阵单行元素相加，最后获得的p1Denom为整个文档集中所有词汇出现的总次数（一个常数）
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    #获得由每个单词 出现频率 组成的矩阵向量 p1Vect
    # p1Vect =p1Num/p1Denom
    # p0Vect =p0Num/p0Denom

    # 由于太多很小的数字相乘，造成 下溢出
    # 解决办法是对乘积取自然对数，通过求对数可以避免下溢出或者浮点数舍入导致的错误
    # 采用自然对数进行处理不会造成任何损失
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)

    '''
    p0Vect: 非侮辱性文档中，每个词出现的概率
    p1Vect: 侮辱性文档中，每个词出现的概率
    pAbusive: 任意一篇文档，是侮辱性文档的概率
    '''
    return p0Vect,p1Vect,pAbusive

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWord2Vec(myVocabList,postinDoc))
# print('--------------')
# print(trainMat)

p0v,p1v,pAb = trainNB0(trainMat,listClasses)

# print(p0v)
# print(p0v[index_stupid])
# print('------------------------')
#
# print(p1v)
# print(p1v[index_stupid])
# print('------------------------')
#
# print(pAb)
# print('------------------------')


# 朴素贝叶斯分类函数（伯努利贝叶斯）
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    """
    
    :param vec2Classify: 要分类的向量，注意向量与列表！！，这个就是numpy形式
    :param p0Vec: 
    :param p1Vec: 
    :param pClass1: 
    :return: 
    """
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    # print(p1,p0)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():

    # 训练部分
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    # 测试部分

    # 输入的测试文档
    testEntry = ['love', 'my', 'dalmation']

    # 将 测试文档 根据 词汇表 转化为 矩阵向量
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))


# testingNB()



def bagOfWords2VecMN(vocabList, inputSet):
    # 创建一个其中所含元素都为 0 的向量，长度与词汇表相同
    returnVec = [0] * len(vocabList)

    # 遍历文档中所有的单词
    for word in inputSet:

        # 如果出现了词汇表中的单词
        if word in vocabList:

            # 将输出的文档向量中的对应值设置为 1
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):
    """
    接受一个 bigString 长字符串，并将其解析为 由长度大于 2 单词 组成的 list
    :param bigString: 长字符串
    :return: 单词组成的 list
    """
    import re

    # 以 [a-zA-Z0-9] 以外的元素进行 拆分
    listOfTokens = re.split('\W+', bigString)

    # 将长度大于 2 的单词转换为小写，并存入 list
    return [tok.lower for tok in listOfTokens if len(tok) > 2]

def spamTest():
    """
    对贝叶斯辣鸡邮件分类器进行自动化处理
    :return:
    """
    # 初始化 文档 list， list 中的每一个元素都是一个 文档（由单词组成的 list）
    docList = []

    # 初始化 文档分类 list， classList 与 docList 中的每个元素 一一对应，即为对应 文档的分类
    classList = []

    # 初始化 全部文本 list， list 中的每个元素， 为 一个单词
    fullText = []

    # 遍历 spam 和 ham 目录下的各个 txt 文件
    for i in range(1, 26):
        # 打开目录下的一个 文本 ，并对其 进行解析 为 文档
        wordList = textParse(open('email/spam/%d.txt' % i).read())

        # 将文档 append 入 docList 中
        docList.append(wordList)

        # 将文档 extend 到 fullText 后
        fullText.extend(wordList)

        # 在 classList 中 添加 文档对应的分类
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 根据 docList 调用 createVocabList 创建 词汇表
    vocabList = createVocabList(docList)

    # 初始化 trainingSet 训练集，一个长度为 50 的 list
    trainingSet = list(range(50))

    # 初始化 testSet 测试集，为空
    testSet = []

    '''
    随机选择数据的一部分作为训练集，而剩余部分作为测试集的过程称为 留存交叉验证
    '''

    # 重复 10 次
    for i in range(10):
        # 从 0 到 训练集长度，随机选择一个整数，作为 randIndex 随机索引
        randIndex = int(random.uniform(0, len(trainingSet)))

        # 测试集 添加 训练集中随机索引 对应的 元素
        testSet.append(trainingSet[randIndex])

        # 从 训练集 中 删除 随机索引 对应的元素
        del(trainingSet[randIndex])

    # 初始化 训练矩阵
    trainMat = []

    # 初始化 训练分类 list
    trainClasses = []

    # 依次遍历 训练集 中的每个元素， 作为 docIndex 文档索引
    for docIndex in trainingSet:
        # 在 trainMat 训练矩阵中 添加 单词向量 矩阵
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        # 在 trainClasses 训练文档分类中 添加 文档对应的分类
        trainClasses.append(classList[docIndex])

    # 调用 trainNB0 函数，以 trainMat 和 trainClasses 作为输入数据，计算 p0V, p1V, pSpam
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    # 初始化 错误统计
    errorCount = 0

    # 遍历 测试集 中的每个元素 作为 文档索引 docIndex
    for docIndex in testSet:
        # 生成单词向量
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])

        # 如果计算后的分类结果 与 实际分类 不同
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 错误数量 + 1
            errorCount += 1

    # 打印 错误率
    print('the error rate is:', float(errorCount)/len(testSet))



spamTest()