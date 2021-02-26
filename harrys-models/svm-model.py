import os
import numpy as np
import pandas
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc

path1 = os.getcwd() + "/../../data/npy-data/"
path2 = os.getcwd() + "/../dataset/"
res = "360"

trainX = np.load(path1 + "trainSamples-"+res+"-withColour.npy")
trainY = np.load(path2 + "train/trainLabels.npy").astype("int")
testX = np.load(path1 + "testSamples-"+res+"-withColour.npy")
testY = np.load(path2 + "test/testLabels.npy").astype("int")

kernelTypes = ["linear", "poly", "rbf"]

for k in kernelTypes:
    svc = SVC(kernel=k)
    svc.fit(trainX, trainY)

    predY = svc.predict(testX)

    print(k + ": " + str(svc.score(testX, testY)))
    print(classification_report(testY, predY))

    pickle.dump(svc, open(os.getcwd() + "/../../model-pickle-files/" + k + "-SVM-model.pkl", "wb"))

    false_positive_rate, recall, thresholds = roc_curve(testY, predY)
    roc_auc = auc(false_positive_rate, recall)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()