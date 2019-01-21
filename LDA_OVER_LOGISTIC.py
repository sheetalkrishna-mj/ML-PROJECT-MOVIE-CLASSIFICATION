import numpy
import os
import pickle
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression


trainFilePath = 'dataset/data.train'
testFilePath = 'dataset/data.test'
evalFilePath = 'dataset/data.eval.anon'
evalIdFilePath = 'dataset/data.eval.anon.id'
highestFeatureValue = 74481
opFileName = "LDA_OVER_LOGISTIC_Predictions.csv"


def init(filePath, testOrTrain):
    if os.path.isfile(testOrTrain+"pickledData.p"):
        return [pickle.load(open(testOrTrain+"pickledData.p", "rb")), pickle.load(open(testOrTrain+"pickledLabels.p", "rb"))]
    else:
        rowArray = []
        colArray = []
        dataArray = []
        labelsArray = []
        fileObject = open(filePath)
        fileContent = fileObject.readlines()
        for i, row in enumerate(fileContent):
            label = int(row[:1])
            labelsArray.append(label)
            for feature in row[2:].split(' '):
                parts = feature.split(':')
                rowArray.append(i)
                colArray.append(int(parts[0]))
                dataArray.append(int(parts[1]))
        if highestFeatureValue not in colArray:
            rowArray.append(2)
            colArray.append(highestFeatureValue)
            dataArray.append(0)
        matrix = csr_matrix((dataArray, (rowArray, colArray)))
        pickle.dump(matrix, open(testOrTrain+"pickledData.p", "wb"))
        pickle.dump(labelsArray, open(testOrTrain+"pickledLabels.p", "wb"))
        return [matrix, labelsArray]


trainMatrixAndLabels = init(trainFilePath, 'train')
trainMatrix = trainMatrixAndLabels[0]
trainLabels = trainMatrixAndLabels[1]
testMatrixAndLabels = init(testFilePath, 'test')
testMatrix = testMatrixAndLabels[0]
testLabels = testMatrixAndLabels[1]
evalMatrix = init(evalFilePath, 'eval')[0]

for treeCount in range(1, 201):
    print("Making and storing: " + str(treeCount))
    tenPercent = indexForShuffling = [numpy.random.choice(trainMatrix.shape[0], int(0.2 * trainMatrix.shape[0]), False)][0]
    tempTrain = trainMatrix[tenPercent, :]
    tempTrainLabels = numpy.array(trainLabels)[tenPercent]
    clf = LogisticRegression(C=0.1).fit(tempTrain, tempTrainLabels)
    tempPredictions = clf.predict(trainMatrix)
    pickle.dump(tempPredictions, open("LogisticTempsTrain/prediction"+str(treeCount)+".p", "wb"))
    tempPredictions = clf.predict(testMatrix)
    pickle.dump(tempPredictions, open("LogisticTempsTest/prediction" + str(treeCount) + ".p", "wb"))
    tempPredictions = clf.predict(evalMatrix)
    pickle.dump(tempPredictions, open("LogisticTempsEval/prediction" + str(treeCount) + ".p", "wb"))

newTrainForSVM = {}
for innerTreeCount in range(1, 201):
    predictionsForTreeCount = pickle.load(open("LogisticTempsTrain/prediction"+str(innerTreeCount)+".p", "rb"))
    if innerTreeCount == 1:
        newTrainForSVM = numpy.array(predictionsForTreeCount)
        newTrainForSVM = newTrainForSVM[numpy.newaxis]
        newTrainForSVM = newTrainForSVM.T
    else:
        innerTemp = numpy.array(predictionsForTreeCount)
        innerTemp = innerTemp[numpy.newaxis]
        newTrainForSVM = numpy.hstack((newTrainForSVM, innerTemp.T))
newTrainForSVM = numpy.asmatrix(newTrainForSVM)
pickle.dump(newTrainForSVM, open("NewTrainForSVMPickle.p", "wb"))

newTestForSVM = {}
for innerTreeCount in range(1, 201):
    predictionsForTreeCount = pickle.load(open("LogisticTempsTest/prediction"+str(innerTreeCount)+".p", "rb"))
    if innerTreeCount == 1:
        newTestForSVM = numpy.array(predictionsForTreeCount)
        newTestForSVM = newTestForSVM[numpy.newaxis]
        newTestForSVM = newTestForSVM.T
    else:
        innerTemp = numpy.array(predictionsForTreeCount)
        innerTemp = innerTemp[numpy.newaxis]
        newTestForSVM = numpy.hstack((newTestForSVM, innerTemp.T))
newTestForSVM = numpy.asmatrix(newTestForSVM)
pickle.dump(newTestForSVM, open("NewTestForSVMPickle.p", "wb"))

newEvalForSVM = {}
for innerTreeCount in range(1, 201):
    predictionsForTreeCount = pickle.load(open("LogisticTempsEval/prediction"+str(innerTreeCount)+".p", "rb"))
    if innerTreeCount == 1:
        newEvalForSVM = numpy.array(predictionsForTreeCount)
        newEvalForSVM = newEvalForSVM[numpy.newaxis]
        newEvalForSVM = newEvalForSVM.T
    else:
        innerTemp = numpy.array(predictionsForTreeCount)
        innerTemp = innerTemp[numpy.newaxis]
        newEvalForSVM = numpy.hstack((newEvalForSVM, innerTemp.T))
newEvalForSVM = numpy.asmatrix(newEvalForSVM)
pickle.dump(newEvalForSVM, open("NewEvalForSVMPickle.p", "wb"))

clf = LDA()
clf.fit(newTrainForSVM, trainLabels)
predictions = clf.predict(newTestForSVM)
predictions = numpy.array(predictions)
testLabels = numpy.array(testLabels)
accuracy = (predictions == testLabels).mean() * 100
predictions = clf.predict(newEvalForSVM)
predictions = numpy.array(predictions)
opFile = open(opFileName, "w")
opFile.write("example_id,label\n")
idFile = open(evalIdFilePath, "r")
ids = idFile.readlines()
for rowNum, prediction in enumerate(predictions):
    prediction = prediction if prediction == 1 else 0
    stuffToWrite = ids[rowNum].replace(' ', '').replace('\n', '') + ',' + str(prediction) + '\n'
    opFile.write(stuffToWrite)
