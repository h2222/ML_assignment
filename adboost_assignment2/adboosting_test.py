from numpy import *
from csv import reader


# 预测分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    aa = dataMatrix[:, dimen].tolist()
    # print(aa)
    for (index, item) in enumerate(aa):
        # print(index)
        # print(item)
        if threshIneq == 'lt':
            if float('%.6f' % float(item[0])) < threshVal:  # 比阀值小，就归为-1
                retArray[index] = -1.0
        elif threshIneq == 'gt':
            if float('%.6f' % float(item[0])) > threshVal:
                retArray[index] = -1.0
    return retArray


# 建立单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf

    #  不同的feature vector循环
    for i in range(n):
        rangeMin = float('%.6f' % min(dataMatrix[:, i]))
        rangeMax = float('%.6f' % max(dataMatrix[:, i]))
        stepSize = (rangeMax - rangeMin) / numSteps
        stepSize = float('%.6f' % stepSize)
        # 不同的分割点为主(thresh value) 循环
        for j in range(-1, int(numSteps) + 1):

            # 大于分割点和小于分割点循环
            for inequal in ['lt', 'gt']:  # less than 和greater than
                # in begin we have :
                #   threshVal  ,  rangeMin,   .........   rangMax
                #          stepSize         10*stepSize
                # round 保留两位小数
                threshVal = float('%.6f' % float(rangeMin + float(j) * stepSize))
                # print('thresh value :', threshVal)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                for k in range(m):
                    # print(predictedVals[k, 0], labelMat[k, 0])
                    if predictedVals[k, 0] == labelMat[k, 0]:
                        errArr[k] = 0  # 分类错误的标记为1，正确为0
                weightedError = D.T * errArr  # 增加分类错误的权重
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" \
                # % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# 训练分类器
def adaBoostTrainDS(dataArr, classLabels, numIt=10):
    errorRateList = []
    weakClassArr = []
    m = shape(dataArr)[0]
    # D 为 sample 权重, 需要更新 初始为 1矩阵 (300 x 1) (value 全都为 1/m)
    D = mat(ones((m, 1)) / m)  # 设置一样的初始权重值
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # classEst 为 G(x) 为若分类器的预测
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 得到“单层”最优决策树
        # print("D:", D.T)
        # α 为分类器权重
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 计算alpha值
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 存储弱分类器
        # print("classEst: ", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))  # 更新分类器权重
        D = D / D.sum()  # 保证权重加和为1
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 检查分类出错的类别
        errorRate = (aggErrors.sum() / m)
        # print("total error: ", error Rate)

        print('numItr: {0}, total error : {1}'.format(i, round(errorRate, 4)))
        errorRateList.append(round(errorRate, 4))
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst, errorRateList


# 用训练出的分类器来作预测
def adaClassify(datToClass, classifierArr, test_label):
    errorRateList = []
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        aggErrors = multiply(sign(aggClassEst) != mat(test_label).T, ones((m, 1)))
        errorRate = (aggErrors.sum() / m)
        errorRateList.append(errorRate)
        print(' test: numItr: {0}, total error : {1}'.format(i, round(errorRate, 4)))

    return sign(aggClassEst), errorRateList


def plotF(ErrorRateList, title='unk', xlabel='unk', ylabel='unk',
          ylim=0.1):
    import matplotlib.pyplot as plt
    length = len(ErrorRateList)
    numIterList = [x for x in range(length)]
    plt.plot(numIterList, ErrorRateList, 'ro-')
    plt.xlim(0, length)
    plt.ylim(0, ylim)
    plt.xticks([x for x in range(length) if x % 5 == 0], [x for x in numIterList if x % 5 == 0], rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    # print(numIterList)


def one_hot(label):
    """
    :param label: [[M],[B], ...]
    :return: [[1],[0], ...]  training label size:(300x1),  test label size:(269x1)
    """
    for i in range(0, label.size):
        if label[i] == 'B':
            label[i] = -1
        else:
            label[i] = 1
    return label


def load_data(file_name):
    with open(file_name, 'rt', encoding='UTF-8') as raw_data:
        readers = reader(raw_data, delimiter=',')
        dataset = array(list(readers))

    train_set = dataset[:300, 2:]
    test_set = dataset[300:, 2:]

    train_set_y = one_hot(dataset[:300, 1:2]).reshape(-1)
    test_set_y = one_hot(dataset[300:, 1:2]).reshape(-1)

    train_set_y = array(train_set_y, dtype='float_').tolist()
    test_set_y = array(test_set_y, dtype='float_').tolist()

    return train_set, train_set_y, test_set, test_set_y


if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y = load_data('wdbc_data.csv')

    train_set_x = mat(train_set_x)
    test_set_x = mat(test_set_x)

    train_set_y = array(train_set_y, dtype='float_').tolist()
    test_set_y = array(test_set_y, dtype='float_').tolist()

    # print('print train set:{0} \n print train set label:{1}'.format(train_set, train_set_y))
    # print('print test set:{0} \n print test set label:{1}'.format(test_set, test_set_y))

    weakClassArr, aggClassEst, errorRateList = adaBoostTrainDS(train_set_x, train_set_y, 50)

    plotF(errorRateList,
          title='relationship between numIter - ErrorRate (train set)',
          xlabel='Number of iteration',
          ylabel='Error Rate')

    predictLabelList, TestErrorRateList = adaClassify(test_set_x, weakClassArr, test_set_y)

    plotF(TestErrorRateList,
          title='relationship between numIter - ErrorRate (test set)',
          xlabel='Number of iteration',
          ylabel='Error Rate')

    print(predictLabelList)
