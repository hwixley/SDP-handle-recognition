import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import cPickle

k = "linear"
path = os.getcwd() + "/../dataset/"
res = "360"

trainX = np.load(path + "training/trainSamples"+res+".npy")
trainY = np.load(path + "training/trainLabels.npy").astype("int")
testX = np.load(path + "testing/testSamples"+res+".npy")
testY = np.load(path + "testing/testLabels.npy").astype("int")

svc = SVC(kernel=k)
svc.fit(trainX, trainY)

predY = svc.predict(testX)

print(k)
print(classification_report(testY, predY))

cPickle.dump(svc, open(k+"-SVM-model.pkl", "wb"))