import os
import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from PIL import Image
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

path1 = os.getcwd() + "/../../data/npy-data/"
path2 = os.getcwd() + "/../dataset/"
res = "360"

trainX = np.load(path1 + "trainSamples-"+res+"-withColour.npy")
trainY = np.load(path2 + "train/trainLabels.npy").astype("int")
testX = np.load(path1 + "testSamples-"+res+"-withColour.npy")
testY = np.load(path2 + "test/testLabels.npy").astype("int")

logR = LogisticRegression().fit(trainX,trainY)
predY = logR.predict(testX)

print(classification_report(testY, predY))

pickle.dump(logR, open("logR-model.pkl","wb"))

false_positive_rate, recall, thresholds = roc_curve(testY,predY)
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