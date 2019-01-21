import random
import time
import pickle
import os.path
import numpy
from scipy.sparse import csr_matrix
import math
import itertools
import matplotlib.pyplot as plt

foldArray = [0, 1, 2, 3, 4]
trainFilePath = 'dataset/data.train'
testFilePath = 'dataset/data.test'
evalFilePath = 'dataset/data.eval.anon'
testIdFilePath = 'dataset/data.test.id'
evalIdFilePath = 'dataset/data.eval.anon.id'
# trainFilePath = 'dataset/faketrain.txt'
# testFilePath = 'dataset/faketest.txt'
# evalFilePath = 'dataset/fakeeval.txt'
# testIdFilePath = 'dataset/testid.txt'
# evalIdFilePath = 'dataset/evalid.txt'
rates = [0, 0.1, 0.01]


def init(filePath):
    """Returns an array of data and labels"""
    if os.path.isfile("pickledData.p"):
        return [pickle.load(open("pickledData.p", "rb")), pickle.load(open("pickledLabels.p", "rb"))]
    else:
        rowArray = []
        colArray = []
        dataArray = []
        labelsArray = []
        fileObject = open(filePath)
        fileContent = fileObject.readlines()
        for i, row in enumerate(fileContent):
            rowArray.append(i)
            colArray.append(0)
            dataArray.append(1)
            label = int(row[:1]) * 2 - 1
            labelsArray.append(label)
            for feature in row[2:].split(' '):
                parts = feature.split(':')
                rowArray.append(i)
                colArray.append(int(parts[0]))
                dataArray.append(int(parts[1]))
        matrix = csr_matrix((dataArray, (rowArray, colArray)))
        pickle.dump(matrix, open("pickledData.p", "wb"))
        pickle.dump(labelsArray, open("pickledLabels.p", "wb"))
        return [matrix, labelsArray]


def trainPerceptron(indexes, innerTrainMatrixAndLabels, innerSimpleWeightVector, innerAverageWeightVector):
    """Trains and returns the weight vector"""
    for index in indexes:
        row = innerTrainMatrixAndLabels[0].getrow(index).toarray()[0]
        label = innerTrainMatrixAndLabels[1][index]
        for rate in rates:
            predictedLabel = innerSimpleWeightVector[rate].dot(row)
            if label * predictedLabel <= 0:
                innerSimpleWeightVector[rate] += rate * label * row
        for rate in rates:
            innerAverageWeightVector[rate] += innerSimpleWeightVector[rate]
    return [innerSimpleWeightVector, innerAverageWeightVector]


def test(innerWeightVector, filePath, fileName, idfile):
    """Tests and returns the accuracy"""
    file1 = open(fileName, "w")
    file1.write("example_id,label\n")
    file2 = open(idfile)
    innerAccuracy = 0
    count = 0
    fileObject = open(filePath)
    fileContent = fileObject.readlines()
    ids = file2.readlines()
    for i, row in enumerate(fileContent):
        rowDs = numpy.zeros(len(innerWeightVector))
        rowDs[0] = 1
        label = int(row[:1]) * 2 - 1
        for feature in row[2:].split(' '):
            parts = feature.split(':')
            rowDs[int(parts[0])] = int(parts[1])
        predictedLabel = innerWeightVector.dot(rowDs)
        predictedLabelToPrint = 1 if predictedLabel > 0 else 0
        stuffToWrite = ids[i].replace(' ', '').replace('\n', '') + ',' + str(predictedLabelToPrint) + '\n'
        file1.write(stuffToWrite)
        if predictedLabel * label > 0:
            innerAccuracy += 1
        count += 1
    file1.close()
    return (innerAccuracy / float(count)) * 100


def getSplits(length):
    """Gets the different folds for cross validation"""
    setSize = math.ceil(length / 5)
    result = []
    setNums = list(range(0, setSize, 1))
    result.append(setNums)
    setNums = list(range(setSize, setSize * 2, 1))
    result.append(setNums)
    setNums = list(range(setSize * 2, setSize * 3, 1))
    result.append(setNums)
    setNums = list(range(setSize * 3, setSize * 4, 1))
    result.append(setNums)
    setNums = list(range(setSize * 4, length, 1))
    result.append(setNums)
    return result


def findSubsets(array, subsetSize):
    """Returns [[1,2,3,4],[0,1,2,3]...]"""
    return set(itertools.combinations(array, subsetSize))


def testForCV(indexes, innerTrainMatrixAndLabels, innerSimpleWeightVector, innerAverageWeightVector):
    """Test for cross-validation"""
    innerAccuracyForSimplePerceptron = {}
    innerAccuracyForAveragePerceptron = {}
    for rate in rates:
        innerAccuracyForSimplePerceptron[rate] = 0
        innerAccuracyForAveragePerceptron[rate] = 0
    count = 0
    for index in indexes:
        row = innerTrainMatrixAndLabels[0].getrow(index).toarray()[0]
        label = innerTrainMatrixAndLabels[1][index]
        for rate in rates:
            predictedLabelBySimple = innerSimpleWeightVector[rate].dot(row)
            predictedLabelByAverage = innerAverageWeightVector[rate].dot(row)
            if label * predictedLabelBySimple > 0:
                innerAccuracyForSimplePerceptron[rate] += 1
            if label * predictedLabelByAverage > 0:
                innerAccuracyForAveragePerceptron[rate] += 1
        count += 1
    for rate in rates:
        innerAccuracyForSimplePerceptron[rate] = (innerAccuracyForSimplePerceptron[rate] / float(count)) * 100
        innerAccuracyForAveragePerceptron[rate] = (innerAccuracyForAveragePerceptron[rate] / float(count)) * 100
    return [innerAccuracyForSimplePerceptron, innerAccuracyForAveragePerceptron]


######################################################
"""1) Read data into matrices/arrays, split into 6 (Including dev dataset"""
######################################################
t0 = time.time()
trainMatrixAndLabels = init(trainFilePath)
splits = getSplits(int(len(trainMatrixAndLabels[1])))  # -len(trainMatrixAndLabels[1])/5))
combinations = list(findSubsets(foldArray, 4))
indexesToTrain = []
weightVectorsForSimplePerceptron = {}
weightVectorsForAveragePerceptron = {}
bestRateForSimplePerceptron = 0
bestRateForAveragePerceptron = 0
bestAccuracyForSimplePerceptron = 0
bestAccuracyForAveragePerceptron = 0
# bestCombiForSimplePerceptron = []
# bestCombiForAveragePerceptron = []

#print("FINISHED 1\n")
######################################################
"""Performing CV to get best learning rate"""
######################################################
acc1 = {}
acc2 = {}
bestEpochForCombi = {}
for rate in rates:
    acc1[rate] = []
    acc2[rate] = []
for combi in combinations:
    indexesToTrain = []
    for i, rate in enumerate(rates):
        numpy.random.seed(i)
        weightVectorsForSimplePerceptron[rate] = numpy.random.uniform(low=-0.01, high=0.01, size=(
            len(trainMatrixAndLabels[0].getrow(0).toarray()[0])))
        numpy.random.seed(i)
        weightVectorsForAveragePerceptron[rate] = numpy.random.uniform(low=-0.01, high=0.01, size=(
            len(trainMatrixAndLabels[0].getrow(0).toarray()[0])))
    for number in combi:
        indexesToTrain += splits[number]
    indexesToTest = splits[list(set(foldArray).difference(set(combi)))[0]]
    for epoch in range(1, 21):
        random.seed(epoch)
        random.shuffle(indexesToTrain)
        temp = trainPerceptron(indexesToTrain, trainMatrixAndLabels, weightVectorsForSimplePerceptron,
                               weightVectorsForAveragePerceptron)
        weightVectorsForSimplePerceptron = temp[0]
        weightVectorsForAveragePerceptron = temp[1]
    temp = testForCV(indexesToTest, trainMatrixAndLabels, weightVectorsForSimplePerceptron,
                     weightVectorsForAveragePerceptron)
    for rate in rates:
        acc1[rate].append(temp[0][rate])
        acc2[rate].append(temp[1][rate])
for rate in rates:
    if numpy.mean(acc1[rate]) >= bestAccuracyForSimplePerceptron:
        bestAccuracyForSimplePerceptron = temp[0][rate]
        # bestCombiForSimplePerceptron = combi
        bestRateForSimplePerceptron = rate
    if numpy.mean(acc2[rate]) >= bestAccuracyForAveragePerceptron:
        bestAccuracyForAveragePerceptron = temp[1][rate]
        # bestCombiForAveragePerceptron = combi
        bestRateForAveragePerceptron = rate
tempArrayToFindBestDevS = []
tempArrayToFindBestDevA = []
for num in range(0, len(foldArray)):
    x = 0
    y = 0
    for rate in rates:
        x += acc1[rate][num]
        y += acc2[rate][num]
    tempArrayToFindBestDevS.append(x / len(rates))
    tempArrayToFindBestDevA.append(y / len(rates))
bestDevSetForSimple = splits[
    list(set(foldArray).difference(set(combinations[tempArrayToFindBestDevS.index(max(tempArrayToFindBestDevS))])))[0]]
bestDevSetForAverage = splits[
    list(set(foldArray).difference(set(combinations[tempArrayToFindBestDevA.index(max(tempArrayToFindBestDevA))])))[0]]
#print("Here 2\n")

"""Finding best epoch for simple perceptron to get best weight vector"""
indexesToTrain = []
bestEpochForSimplePerceptron = 0
bestEpochForAveragePerceptron = 0
bestAccuracyForSimplePerceptron = 0
bestAccuracyForAveragePerceptron = 0
bestWeightVectorForSimplePerceptron = []
bestWeightVectorForAveragePerceptron = []
for i, rate in enumerate(rates):
    numpy.random.seed(i)
    weightVectorsForSimplePerceptron[rate] = numpy.random.uniform(low=-0.01, high=0.01, size=(
        len(trainMatrixAndLabels[0].getrow(0).toarray()[0])))
    numpy.random.seed(i)
    weightVectorsForAveragePerceptron[rate] = numpy.random.uniform(low=-0.01, high=0.01, size=(
        len(trainMatrixAndLabels[0].getrow(0).toarray()[0])))

for number in combinations[tempArrayToFindBestDevS.index(max(tempArrayToFindBestDevS))]:
    indexesToTrain += splits[number]

# indexesToTest = splits[list(set(foldArray).difference(set(bestCombiForSimplePerceptron)))[0]]
indexesToTest = bestDevSetForSimple
simpleEpochAccuracies = []
for epoch in range(1, 21):
    random.seed(epoch)
    random.shuffle(indexesToTrain)
    temp = trainPerceptron(indexesToTrain, trainMatrixAndLabels, weightVectorsForSimplePerceptron,
                           weightVectorsForAveragePerceptron)
    weightVectorsForSimplePerceptron = temp[0]
    weightVectorsForAveragePerceptron = temp[1]
    temp = testForCV(indexesToTest, trainMatrixAndLabels, weightVectorsForSimplePerceptron,
                     weightVectorsForAveragePerceptron)
    simpleEpochAccuracies.append(temp[0][bestRateForSimplePerceptron])
    if temp[0][bestRateForSimplePerceptron] > bestAccuracyForSimplePerceptron:
        bestEpochForSimplePerceptron = epoch
        bestAccuracyForSimplePerceptron = temp[0][bestRateForSimplePerceptron]
        bestWeightVectorForSimplePerceptron = weightVectorsForSimplePerceptron[bestRateForSimplePerceptron]
indexesToTrain = []
for number in foldArray:
    indexesToTrain += splits[number]
for epoch in range(1, bestEpochForSimplePerceptron + 1):
    random.seed(epoch)
    random.shuffle(indexesToTrain)
    temp = trainPerceptron(indexesToTrain, trainMatrixAndLabels, weightVectorsForSimplePerceptron,
                           weightVectorsForAveragePerceptron)
    weightVectorsForSimplePerceptron = temp[0]
    weightVectorsForAveragePerceptron = temp[1]

bestWeightVectorForSimplePerceptron = weightVectorsForSimplePerceptron[bestRateForSimplePerceptron]
#print("Here 3\n")
"""Finding best epoch for average perceptron to get best weight vector"""
indexesToTrain = []
averageEpochAccuracies = []
for number in combinations[tempArrayToFindBestDevA.index(max(tempArrayToFindBestDevA))]:
    indexesToTrain += splits[number]
indexesToTest = bestDevSetForAverage
for i, rate in enumerate(rates):
    numpy.random.seed(i)
    weightVectorsForSimplePerceptron[rate] = numpy.random.uniform(low=-0.01, high=0.01, size=(
        len(trainMatrixAndLabels[0].getrow(0).toarray()[0])))
    numpy.random.seed(i)
    weightVectorsForAveragePerceptron[rate] = numpy.random.uniform(low=-0.01, high=0.01, size=(
        len(trainMatrixAndLabels[0].getrow(0).toarray()[0])))

# for number in bestCombiForAveragePerceptron:
# 	indexesToTrain += splits[number]
# indexesToTest = splits[list(set(foldArray).difference(set(bestCombiForAveragePerceptron)))[0]]

for epoch in range(1, 21):
    random.seed(epoch)
    random.shuffle(indexesToTrain)
    temp = trainPerceptron(indexesToTrain, trainMatrixAndLabels, weightVectorsForSimplePerceptron,
                           weightVectorsForAveragePerceptron)
    weightVectorsForSimplePerceptron = temp[0]
    weightVectorsForAveragePerceptron = temp[1]
    temp = testForCV(indexesToTest, trainMatrixAndLabels, weightVectorsForSimplePerceptron,
                     weightVectorsForAveragePerceptron)
    averageEpochAccuracies.append(temp[1][bestRateForAveragePerceptron])
    if temp[1][bestRateForAveragePerceptron] > bestAccuracyForAveragePerceptron:
        bestEpochForAveragePerceptron = epoch
        bestAccuracyForAveragePerceptron = temp[0][bestRateForAveragePerceptron]
        bestWeightVectorForAveragePerceptron = weightVectorsForAveragePerceptron[bestRateForAveragePerceptron]

indexesToTrain = []
for number in foldArray:
    indexesToTrain += splits[number]
for epoch in range(1, bestEpochForAveragePerceptron + 1):
    random.seed(epoch)
    random.shuffle(indexesToTrain)
    temp = trainPerceptron(indexesToTrain, trainMatrixAndLabels, weightVectorsForSimplePerceptron,
                           weightVectorsForAveragePerceptron)
    weightVectorsForSimplePerceptron = temp[0]
    weightVectorsForAveragePerceptron = temp[1]

bestWeightVectorForAveragePerceptron = weightVectorsForAveragePerceptron[bestRateForAveragePerceptron]
accuracySimple = test(bestWeightVectorForSimplePerceptron, testFilePath, "output.txt", testIdFilePath)
accuracyAverage = test(bestWeightVectorForAveragePerceptron, testFilePath, "output.txt", testIdFilePath)

print("Accuracy on test dataset for Simple Perceptron: " + str(accuracySimple) + " and Best Rate is: " + str(
    bestRateForSimplePerceptron))
print("Accuracy on test dataset for Average Perceptron: " + str(accuracyAverage) + " and Best Rate is: " + str(
    bestRateForAveragePerceptron))
test(bestWeightVectorForSimplePerceptron, evalFilePath, "SimplePredictions.csv", evalIdFilePath)
test(bestWeightVectorForAveragePerceptron, evalFilePath, "AveragePredictions.csv", evalIdFilePath)

#print(simpleEpochAccuracies)
#print(averageEpochAccuracies)

plt.plot(averageEpochAccuracies)
plt.xticks(list(range(1, 21, 1)))
plt.xlabel('Epoch Count')
plt.ylabel('Accuracy')
plt.title('Average')
plt.show()
plt.plot(simpleEpochAccuracies)
plt.xticks(list(range(1, 21, 1)))
plt.xlabel('Epoch Count')
plt.ylabel('Accuracy')
plt.title('Simple')
plt.show()

"""Getting predictions to report"""
t1 = time.time()
print('Time taken: ' + str((t1 - t0) / 60) + ' min(s)')

