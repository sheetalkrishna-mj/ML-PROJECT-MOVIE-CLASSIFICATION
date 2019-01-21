import numpy
import pandas
import copy
import itertools
import matplotlib.pyplot as plt
import os
import pickle
import time
import math

differentLearningRates = [0.01, 0.001, 0.0001]
differentRegParams = [10000, 1000, 100]  # , 10, 1, 0.1 , 0.01, 0.001, 0.0001, 0.00001]
foldArray = [0, 1, 2, 3, 4]
seed = 0
foldFilePath = 'dataset/CVSplits/training0'
trainFilePath = 'dataset/data.train'
testFilePath = 'dataset/data.test'
evalFilePath = 'dataset/data.eval.anon'
evalIdFilePath = 'dataset/data.eval.anon.id'
highestFeatureValue = 74481
# testIdFilePath = 'dataset/data.test.id'
opFileName = "SVM_Predictions.csv"
epochsForQ1 = 20
epochsForQ2 = 30


def init(filePath, testOrTrain, multiplier=999):
    if multiplier != 999:
        multiplier *= 5000
    else:
        multiplier = 0
    fileObject = open(filePath)
    fileContent = fileObject.readlines()
    data = {}
    labelsArray = []
    for i, row in enumerate(fileContent):
        data[i + multiplier] = {}
        data[i + multiplier][0] = 1
        splitRow = row.replace('\n', '').split(' ')
        labelsArray.append(int(splitRow[0]) * 2 - 1)
        for feature in splitRow[1:]:
            parts = feature.split(':')
            data[i + multiplier][int(parts[0])] = int(parts[1])
    return [data, labelsArray]


# noinspection PyBroadException
def trainSVM(curEpoch, matrix, trainLbls, wtVector, innerIndexes):
    if curEpoch == 1:
        wtVector = {}
        for innerRate in differentLearningRates:
            wtVector[innerRate] = {}
            for innerRegParam in differentRegParams:
                numpy.random.seed(seed)
                wtVector[innerRate][innerRegParam] = numpy.zeros(highestFeatureValue + 1)
    for rowNumber in innerIndexes:
        dummyRow = numpy.zeros(highestFeatureValue + 1)
        for feature in matrix[rowNumber]:
            dummyRow[feature] = matrix[rowNumber][feature]
        dummyRow = numpy.array(dummyRow)
        for innerRate in differentLearningRates:
            newRate = innerRate / float(1 + curEpoch)
            for innerRegParam in differentRegParams:
                predictedLabel = wtVector[innerRate][innerRegParam].dot(dummyRow)
                if predictedLabel * trainLbls[rowNumber] <= 1:
                	wtVector[innerRate][innerRegParam] = (1 - newRate) * wtVector[innerRate][innerRegParam] + newRate * innerRegParam * trainLbls[rowNumber] * dummyRow
                else:
                	wtVector[innerRate][innerRegParam] = (1 - newRate) * wtVector[innerRate][innerRegParam]
    return wtVector


def test(matrix, testLbls, wtVector, returnPredictions=0):
    innerAccuracy = {}
    innerPredictions = {}
    for innerRate in differentLearningRates:
        innerAccuracy[innerRate] = {}
        innerPredictions[innerRate] = {}
        for innerRegParam in differentRegParams:
            innerAccuracy[innerRate][innerRegParam] = {}
            innerPredictions[innerRate][innerRegParam] = {}
            innerAccuracy[innerRate][innerRegParam]['Accuracy'] = 0
            innerAccuracy[innerRate][innerRegParam]['TP'] = 0
            innerAccuracy[innerRate][innerRegParam]['FP'] = 0
            innerAccuracy[innerRate][innerRegParam]['FN'] = 0
            # innerAccuracy[innerRate][regParam]['TN'] = 0
            innerAccuracy[innerRate][innerRegParam]['F1'] = 0
            innerAccuracy[innerRate][innerRegParam]['P'] = 0
            innerAccuracy[innerRate][innerRegParam]['R'] = 0
            innerPredictions[innerRate][innerRegParam]['Predictions'] = []
    if returnPredictions == 0:
        for rowNumber in matrix:
            dummyRow = numpy.zeros(highestFeatureValue + 1)
            for feature in matrix[rowNumber]:
                dummyRow[feature] = matrix[rowNumber][feature]
            
            for innerRate in differentLearningRates:
                for innerRegParam in differentRegParams:
                    predictedLabel = 1 if ((wtVector[innerRate][innerRegParam].dot(dummyRow)) > 0) else -1
                    if predictedLabel == 1 and testLbls[rowNumber] == 1:
                        innerAccuracy[innerRate][innerRegParam]['TP'] += 1
                    if predictedLabel == 1 and testLbls[rowNumber] == -1:
                        innerAccuracy[innerRate][innerRegParam]['FP'] += 1
                    if predictedLabel == -1 and testLbls[rowNumber] == 1:
                        innerAccuracy[innerRate][innerRegParam]['FN'] += 1
                    # if predictedLabel == -1 and testLbls[k] == -1:
                    #     innerAccuracy[innerRate][regParam]['TN'] += 1
                    if predictedLabel == testLbls[rowNumber]:
                        innerAccuracy[innerRate][innerRegParam]['Accuracy'] += 1
  
                        

        totalRows = float(len(matrix))
        for innerRate in differentLearningRates:
            for innerRegParam in differentRegParams:
                innerAccuracy[innerRate][innerRegParam]['Accuracy'] = (innerAccuracy[innerRate][innerRegParam][
                                                                           'Accuracy'] / float(totalRows)) * 100
                if (innerAccuracy[innerRate][innerRegParam]['TP'] + innerAccuracy[innerRate][innerRegParam]['FP']) > 0:
                    precision = (innerAccuracy[innerRate][innerRegParam]['TP']) / float(
                        innerAccuracy[innerRate][innerRegParam]['TP'] + innerAccuracy[innerRate][innerRegParam]['FP'])
                    innerAccuracy[innerRate][innerRegParam]['P'] = precision
                else:
                    precision = 0
                    innerAccuracy[innerRate][innerRegParam]['P'] = precision
                if (innerAccuracy[innerRate][innerRegParam]['TP'] + innerAccuracy[innerRate][innerRegParam]['FN']) > 0:
                    recall = (innerAccuracy[innerRate][innerRegParam]['TP']) / float(
                        innerAccuracy[innerRate][innerRegParam]['TP'] + innerAccuracy[innerRate][innerRegParam]['FN'])
                    innerAccuracy[innerRate][innerRegParam]['R'] = recall
                else:
                    recall = 0
                    innerAccuracy[innerRate][innerRegParam]['R'] = recall
                if precision + recall > 0:
                    innerAccuracy[innerRate][innerRegParam]['F1'] = 2 * (precision * recall) / float(precision + recall)
                else:
                    innerAccuracy[innerRate][innerRegParam]['F1'] = 0
        return innerAccuracy
    else:
        for rowNumber in matrix:
            dummyRow = numpy.zeros(highestFeatureValue + 1)
            for feature in matrix[rowNumber]:
                dummyRow[feature] = matrix[rowNumber][feature]
            for innerRate in differentLearningRates:
                for innerRegParam in differentRegParams:
                    predictedLabel = 1 if ((wtVector[innerRate][innerRegParam].dot(dummyRow)) > 0) else -1
                    innerPredictions[innerRate][innerRegParam]['Predictions'].append(predictedLabel)
        return innerPredictions


def findSubsets(S, m):
    return set(itertools.combinations(S, m))


def getWV(innerFoldCount):
    print("Getting weight vector for combi: " + innerFoldCount)
    if os.path.isfile("pickledWeightVectorForFoldCombi" + innerFoldCount + ".p"):
        return pickle.load(open("pickledWeightVectorForFoldCombi" + innerFoldCount + ".p", "rb"))
    else:
        return {}


# CROSS VALIDATION TO FIND BEST HYPER-PARAMETERS
t0 = time.time()
combinations = list(findSubsets(foldArray, 4))
allAccuracies = pandas.DataFrame(
    data={'LearningRate': [], 'Regularization': [], 'Accuracy': [], 'F1': [], 'P': [], 'R': []})
for foldCount, fold in enumerate(combinations):
    tempMatrix = {}
    tempLabels = []
    width = 0
    for num, number in enumerate(fold):
        trainMatrixAndLabels = init(foldFilePath + str(number) + '.data', foldFilePath + str(number), num)
        trainMatrix = trainMatrixAndLabels[0]
        trainLabels = trainMatrixAndLabels[1]
        if len(tempMatrix) == 0:
            tempMatrix = trainMatrix.copy()
        else:
            tempMatrix = {**tempMatrix, **trainMatrix}
        tempLabels += trainLabels
    trainMatrix = tempMatrix
    trainLabels = numpy.array(tempLabels)
    testMatrixAndLabels = init(foldFilePath + str(list(set(foldArray).difference(set(fold)))[0]) + ".data",
                               foldFilePath + str(list(set(foldArray).difference(set(fold)))[0]))
    testMatrix = testMatrixAndLabels[0]
    testLabels = testMatrixAndLabels[1]

    #  weightVector = {}
    stepCount = 0
    weightVector = getWV(str(foldCount))
    if weightVector == {}:
        for ep in range(1, epochsForQ1 + 1):
            print("CV Epoch: " + str(ep) + "\n")
            numpy.random.seed(ep)
            indexForShuffling = [numpy.random.choice(len(trainMatrix), len(trainMatrix), False)][0]
            weightVector = trainSVM(ep, trainMatrix, trainLabels, weightVector, indexForShuffling)
        pickle.dump(weightVector, open("pickledWeightVectorForFoldCombi" + str(foldCount) + ".p", "wb"))
    accuracy = test(testMatrix, testLabels, weightVector)
    for rate in differentLearningRates:
        for regParam in differentRegParams:
            allAccuracies = allAccuracies.append(pandas.DataFrame(
                data={'LearningRate': [rate], 'Regularization': [regParam],
                      'Accuracy': [accuracy[rate][regParam]['Accuracy']], 'F1': [accuracy[rate][regParam]['F1']],
                      'P': [accuracy[rate][regParam]['P']], 'R': [accuracy[rate][regParam]['R']]}), ignore_index=True,
                sort=True)

avgSvm = pandas.DataFrame(data={'LearningRate': [], 'Regularization': [], 'Accuracy': [], 'F1': [], 'P': [], 'R': []})
for rate in differentLearningRates:
    for regParam in differentRegParams:
        averageSvm = allAccuracies[allAccuracies['LearningRate'] == rate][allAccuracies['Regularization'] == regParam]
        avgSvm = avgSvm.append(pandas.DataFrame(
            data={'LearningRate': [rate], 'Regularization': [regParam], 'Accuracy': [averageSvm['Accuracy'].mean()],
                  'F1': [averageSvm['F1'].mean()], 'R': [averageSvm['R'].mean()], 'P': [averageSvm['P'].mean()]}),
            ignore_index=True, sort=True)
bestLearningRateForSvm = avgSvm.loc[avgSvm['Accuracy'].idxmax()].LearningRate
bestRegParamForSvm = avgSvm.loc[avgSvm['Accuracy'].idxmax()].Regularization
tempBestCvF1 = avgSvm.loc[avgSvm['Accuracy'].idxmax()].F1
tempBestCvP = avgSvm.loc[avgSvm['Accuracy'].idxmax()].P
tempBestCvR = avgSvm.loc[avgSvm['Accuracy'].idxmax()].R
tempBestCvAccuracy = avgSvm.loc[avgSvm['Accuracy'].idxmax()].Accuracy

print('__________________________________________________________________________________________\n')
print('The best learning rate after CV for SVM is: ' + str(bestLearningRateForSvm) +
      ' and best Regularization param is: ' + str(bestRegParamForSvm) +
      ' with an average F-score of: ' + str(tempBestCvF1) +
      ' , an average Precision of: ' + str(tempBestCvP) +
      ' , an average Recall of: ' + str(tempBestCvR) +
      ' and an average Accuracy of: ' + str(tempBestCvAccuracy) + "%\n")

if os.path.isfile("pickledBestEpochStuffForSVM.p"):
    bestEpochClassifier = pickle.load(open("pickledBestEpochStuffForSVM.p", "rb"))
    bestSvmClassifier = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].Classifier
    bestSvmEpoch = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].Epoch
    accuracy = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].Accuracy
    F1 = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].F1
    r = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].R
    p = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].P
else:
    tempMatrix = {}
    tempLabels = []
    for num, number in enumerate([0, 1, 2, 3]):
        trainMatrixAndLabels = init(foldFilePath + str(number) + '.data', foldFilePath + str(number), num)
        trainMatrix = trainMatrixAndLabels[0]
        trainLabels = trainMatrixAndLabels[1]
        if len(tempMatrix) == 0:
            tempMatrix = trainMatrix
        else:
            tempMatrix = {**tempMatrix, **trainMatrix}
        tempLabels += trainLabels
    trainMatrix = tempMatrix
    trainLabels = numpy.array(tempLabels)
    devMatrixAndLabels = init(foldFilePath + str(4) + '.data', foldFilePath + str(4))
    devMatrix = devMatrixAndLabels[0]
    devLabels = devMatrixAndLabels[1]
    weightVector = {}
    bestEpochClassifier = pandas.DataFrame(columns=['Epoch', 'Accuracy', 'F1', 'R', 'P', 'Classifier'])
    for ep in range(1, epochsForQ2 + 1):
        print("Is the best Epoch, epoch: " + str(ep) + "?\n")
        numpy.random.seed(ep)
        indexForShuffling = [numpy.random.choice(len(trainMatrix), len(trainMatrix), False)][0]
        trainMatrix = trainMatrix
        trainLabels = trainLabels
        weightVector = trainSVM(ep, trainMatrix, trainLabels, weightVector, indexForShuffling)
        tempTestDetails = test(devMatrix, devLabels, weightVector)[bestLearningRateForSvm][bestRegParamForSvm]
        accuracy = tempTestDetails['Accuracy']
        F1 = tempTestDetails['F1']
        r = tempTestDetails['R']
        p = tempTestDetails['P']
        df = pandas.DataFrame(data={'Epoch': [copy.deepcopy(ep)],
                                    'Accuracy': [copy.deepcopy(accuracy)],
                                    'Classifier': [copy.deepcopy(weightVector)],
                                    'F1': [copy.deepcopy(F1)],
                                    'R': [copy.deepcopy(r)],
                                    'P': [copy.deepcopy(p)]})
        bestEpochClassifier = bestEpochClassifier.append(df, ignore_index=True, sort=True)
    pickle.dump(bestEpochClassifier, open("pickledBestEpochStuffForSVM.p", "wb"))
    bestSvmClassifier = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].Classifier
    bestSvmEpoch = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].Epoch
    accuracy = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].Accuracy
    F1 = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].F1
    r = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].R
    p = bestEpochClassifier.loc[bestEpochClassifier['Accuracy'].idxmax()].P

print('Epoch at which SVM performs best is: ' + str(bestSvmEpoch) +
      ' with Accuracy: ' + str(accuracy) +
      '% , F-Score: ' + str(F1) +
      ' , Recall: ' + str(r) +
      ' and Precision: ' + str(p) + ' \n')

trainMatrixAndLabels = init(trainFilePath, 'train')
trainMatrix = trainMatrixAndLabels[0]
trainLabels = trainMatrixAndLabels[1]
testMatrixAndLabels = init(testFilePath, 'test')
testMatrix = testMatrixAndLabels[0]
testLabels = testMatrixAndLabels[1]
tempTestDetails = test(testMatrix, testLabels, bestSvmClassifier)[bestLearningRateForSvm][bestRegParamForSvm]
accuracy = tempTestDetails["Accuracy"]
F1 = tempTestDetails['F1']
r = tempTestDetails['R']
p = tempTestDetails['P']
print('Accuracy on testing the test dataset with SVM: ' + str(accuracy) +
      '% , F-Score is: ' + str(F1) +
      ', Recall is: ' + str(r) +
      ' and Precision is: ' + str(p) +
      '\n')
print('__________________________________________________________________________________________\n')


evalMatrixAndLabels = init(evalFilePath, 'eval')
evalMatrix = evalMatrixAndLabels[0]
evalLabels = evalMatrixAndLabels[1]
predictions = test(evalMatrix, evalLabels, bestSvmClassifier, 1)[bestLearningRateForSvm][bestRegParamForSvm]['Predictions']
opFile = open(opFileName, "w")
opFile.write("example_id,label\n")
idFile = open(evalIdFilePath, "r")
ids = idFile.readlines()
for rowNum, prediction in enumerate(predictions):
    prediction = prediction if prediction == 1 else 0
    stuffToWrite = ids[rowNum].replace(' ', '').replace('\n', '') + ',' + str(prediction) + '\n'
    opFile.write(stuffToWrite)
t1 = time.time()
print('Time taken: ' + str((t1 - t0) / 60) + ' min(s)')

plt.plot(bestEpochClassifier['Accuracy'])
plt.xlabel('Epoch Count')
plt.ylabel('Accuracy')
plt.title('SVM: Lambda= '+str(bestLearningRateForSvm)+', C='+str(bestRegParamForSvm)+', Best Epoch='+str(bestSvmEpoch))
plt.show()
plt.plot(bestEpochClassifier['F1'])
plt.xlabel('Epoch Count')
plt.ylabel('F-Score')
plt.title('SVM: Lambda= '+str(bestLearningRateForSvm)+', C='+str(bestRegParamForSvm)+', Best Epoch='+str(bestSvmEpoch))
plt.show()

