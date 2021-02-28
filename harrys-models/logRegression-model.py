import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from PIL import Image
import pickle
from sklearn.metrics import roc_curve, auc, plot_roc_curve, roc_auc_score
import matplotlib.pyplot as plt

path1 = os.getcwd() + "/../../data/npy-data/"
path2 = os.getcwd() + "/../dataset/"

testY = np.load(path2 + "test/testLabels.npy").astype("int")

resOptions = ["360", "504", "576"]

for res in resOptions:
    testX = np.load(path1 + "testSamples-" + res + "-withColour.npy")
    logR = pd.read_pickle(os.getcwd()+"/log-reg-model-" + res + ".pkl")

    predY = logR.predict(testX)

    print(classification_report(testY, predY))

    fpr, tpr, thresh = roc_curve(testY, predY)
    auc = roc_auc_score(testY, predY)
    plt.plot(fpr, tpr, label="Logistic Regression "+res+"p, AUC=" + str(auc))



plt.legend(loc=0)
#plot_roc_curve(logR, testX, testY)
plt.title("Receiver Operating Characteristic")

plt.show()