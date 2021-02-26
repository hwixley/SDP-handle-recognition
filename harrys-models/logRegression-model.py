import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

path = os.getcwd() + "/../dataset/"
res = "360"

trainX = np.load(path + "training/trainSamples"+res+".npy")
trainY = np.load(path + "training/trainLabels.npy").astype("int")
testX = np.load(path + "testing/testSamples"+res+".npy")
testY = np.load(path + "testing/testLabels.npy").astype("int")

logR = LogisticRegression().fit(trainX, trainY)
predY = logR.predict(testX)

print(classification_report(testY, predY))