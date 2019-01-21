import numpy
import os
import pickle
import math

bestLearningRateForSvm = 0.0001
bestLearningRateForLogistic = 0.1
bestCForSvm = 1000
bestSigmaForLogistic = 10000
bestEpochForSvm = 15
bestEpochForLogistic = 10

foldArray = [0, 1, 2, 3, 4]
seed = 0
foldFilePath = 'dataset/CVSplits/training0'
trainFilePath = 'dataset/data.train'
testFilePath = 'dataset/data.test'
evalFilePath = 'dataset/data.eval.anon'
highestFeatureValue = 74481
opFileName = "SVMOverLogistic_Predictions.csv"


def init(filePath, filename):
    if os.path.isfile("pickled" + filename + ".p"):
        return [pickle.load(open("pickled" + filename + ".p", "rb")), pickle.load(open("pickled" + filename + "Labels.p", "rb"))]
    fileObject = open(filePath)
    fileContent = fileObject.readlines()
    data = {}
    labelsArray = []
    for i, row in enumerate(fileContent):
        data[i] = {}
        data[i][0] = 1
        splitRow = row.replace('\n', '').split(' ')
        labelsArray.append(int(splitRow[0]) * 2 - 1)
        for feature in splitRow[1:]:
            parts = feature.split(':')
            data[i][int(parts[0])] = int(parts[1])
    pickle.dump(data, open("pickled" + filename + ".p", "wb"))
    pickle.dump(labelsArray, open("pickled" + filename + "Labels.p", "wb"))
    return [data, labelsArray]


trainMatrixAndLabels = init(trainFilePath, "Train")
trainMatrix = trainMatrixAndLabels[0]
trainLabels = trainMatrixAndLabels[1]
testMatrixAndLabels = init(testFilePath, "Test")
testMatrix = testMatrixAndLabels[0]
testLabels = testMatrixAndLabels[1]
evalMatrix = init(evalFilePath, "Eval")[0]


def trainLogistic(matrix, labels, innerIndexes):
    rate = bestLearningRateForLogistic
    wtVector = []
    for curEpoch in range(1, bestEpochForLogistic + 1):
        if curEpoch == 1:
            wtVector = numpy.zeros(highestFeatureValue + 1)
        for rowNumber in innerIndexes:
            dummyRow = numpy.zeros(highestFeatureValue + 1)
            for feature in matrix[rowNumber]:
                dummyRow[feature] = matrix[rowNumber][feature]
            dummyRow = numpy.array(dummyRow)
            try:
                x = labels[rowNumber] * wtVector.dot(dummyRow)
                wtVector = wtVector + ((rate * labels[rowNumber] * dummyRow) / float(1 + math.exp(x))) - ((2 * rate * wtVector) / float(bestSigmaForLogistic))
            except:
                continue
            rate = bestLearningRateForLogistic / float(1 + curEpoch)
    return wtVector


def getPredictions(matrix, wtVector):
    innerPredictions = []
    for rowNumber in matrix:
        dummyRow = numpy.zeros(highestFeatureValue + 1)
        for feature in matrix[rowNumber]:
            dummyRow[feature] = matrix[rowNumber][feature]
        predictedLabel = 1 if ((wtVector.dot(dummyRow)) > 0) else -1
        innerPredictions.append(predictedLabel)
    return innerPredictions


for treeCount in range(1, 201):
    if os.path.isfile("LogisticTempsTestOwn/prediction" + str(treeCount) + ".p"):
        continue
    print("Making and storing: " + str(treeCount))
    numpy.random.seed(treeCount)
    tenPercent = indexForShuffling = [numpy.random.choice(len(trainMatrix), int(0.1 * len(trainMatrix)), False)][0]
    weightVector = trainLogistic(trainMatrix, trainLabels, tenPercent)
    tempTrainPredictions = getPredictions(trainMatrix, weightVector)
    tempTestPredictions = getPredictions(testMatrix, weightVector)
    tempEvalPredictions = getPredictions(evalMatrix, weightVector)
    accuracy = (numpy.array(tempTrainPredictions) == numpy.array(trainLabels)).mean()
    print(accuracy)
    pickle.dump(tempTrainPredictions, open("LogisticTempsTrainOwn/prediction" + str(treeCount) + ".p", "wb"))
    accuracy = (numpy.array(tempTestPredictions) == numpy.array(testLabels)).mean()
    pickle.dump(tempTestPredictions, open("LogisticTempsTestOwn/prediction" + str(treeCount) + ".p", "wb"))
    print(accuracy)
    pickle.dump(tempEvalPredictions, open("LogisticTempsEvalOwn/prediction" + str(treeCount) + ".p", "wb"))

newTrainForSVM = {}
if os.path.isfile("NewTrainForSVMPickle.p"):
    newTrainForSVM = pickle.load(open("NewTrainForSVMPickle.p", "rb"))
else:
    for innerTreeCount in range(1, 201):
        predictionsForTreeCount = pickle.load(open("LogisticTempsTrainOwn/prediction"+str(innerTreeCount)+".p", "rb"))
        if innerTreeCount == 1:
            newTrainForSVM = numpy.array(predictionsForTreeCount)
            newTrainForSVM = newTrainForSVM[numpy.newaxis]
            newTrainForSVM = newTrainForSVM.T
        else:
            innerTemp = numpy.array(predictionsForTreeCount)
            innerTemp = innerTemp[numpy.newaxis]
            newTrainForSVM = numpy.hstack((newTrainForSVM, innerTemp.T))
    bias = numpy.ones(len(newTrainForSVM))
    bias = bias[numpy.newaxis]
    newTrainForSVM = numpy.hstack((bias.T, newTrainForSVM))
    newTrainForSVM = numpy.asmatrix(newTrainForSVM)
    pickle.dump(newTrainForSVM, open("NewTrainForSVMPickle.p", "wb"))

newTestForSVM = {}
if os.path.isfile("NewTestForSVMPickle.p"):
    newTestForSVM = pickle.load(open("NewTestForSVMPickle.p", "rb"))
else:
    for innerTreeCount in range(1, 201):
        predictionsForTreeCount = pickle.load(open("LogisticTempsTestOwn/prediction"+str(innerTreeCount)+".p", "rb"))
        if innerTreeCount == 1:
            newTestForSVM = numpy.array(predictionsForTreeCount)
            newTestForSVM = newTestForSVM[numpy.newaxis]
            newTestForSVM = newTestForSVM.T
        else:
            innerTemp = numpy.array(predictionsForTreeCount)
            innerTemp = innerTemp[numpy.newaxis]
            newTestForSVM = numpy.hstack((newTestForSVM, innerTemp.T))
    bias = numpy.ones(len(newTestForSVM))
    bias = bias[numpy.newaxis]
    newTestForSVM = numpy.hstack((bias.T, newTestForSVM))
    newTestForSVM = numpy.asmatrix(newTestForSVM)
    pickle.dump(newTestForSVM, open("NewTestForSVMPickle.p", "wb"))

newEvalForSVM = {}
if os.path.isfile("NewEvalForSVMPickle.p"):
    newEvalForSVM = pickle.load(open("NewEvalForSVMPickle.p", "rb"))
else:
    for innerTreeCount in range(1, 201):
        predictionsForTreeCount = pickle.load(open("LogisticTempsEvalOwn/prediction"+str(innerTreeCount)+".p", "rb"))
        if innerTreeCount == 1:
            newEvalForSVM = numpy.array(predictionsForTreeCount)
            newEvalForSVM = newEvalForSVM[numpy.newaxis]
            newEvalForSVM = newEvalForSVM.T
        else:
            innerTemp = numpy.array(predictionsForTreeCount)
            innerTemp = innerTemp[numpy.newaxis]
            newEvalForSVM = numpy.hstack((newEvalForSVM, innerTemp.T))
    bias = numpy.ones(len(newEvalForSVM))
    bias = bias[numpy.newaxis]
    newEvalForSVM = numpy.hstack((bias.T, newEvalForSVM))
    newEvalForSVM = numpy.asmatrix(newEvalForSVM)
    pickle.dump(newEvalForSVM, open("NewEvalForSVMPickle.p", "wb"))

import SvmOverLogistic_SVM
