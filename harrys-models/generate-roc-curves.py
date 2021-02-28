import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc, plot_roc_curve, roc_auc_score

log_reg_models = ["360", "504", "576"]
svm_models = ["linear", "poly", "rbf"]
path1 = os.getcwd() + "/../../data/npy-data/"
path2 = os.getcwd() + "/../dataset/"
testY = np.load(path2 + "test/testLabels.npy").astype("int")
rc = np.empty((6, 2))

labels = ['Log. Reg. 360p, AUC=','Log. Reg. 504p, AUC=','Log. Reg. 576p, AUC=','Linear SVM 360p, AUC=',
          'Poly SVM 360p, AUC=','RBF SVM 360p, AUC=']
colours = ["red","green","blue","purple","orange","grey"]


for m in range(3):
    testX = np.load(path1 + "testSamples-" + log_reg_models[m] + "-withColour.npy")

    model = pd.read_pickle(os.getcwd()+ "/log-reg-model-"+log_reg_models[m]+".pkl")
    #plot_roc_curve(model, testX, testY)
    predY = model.predict_proba(testX)

    #print(classification_report(testY, predY))
    pred_y = 0.5 + predY[:,1] - predY[:,0]

    rc = roc_curve(testY, pred_y)
    auc = int(roc_auc_score(testY, pred_y)*100)/100
    plt.plot(rc[0], rc[1], c=colours[m], label=labels[m] + str(auc))

testX = np.load(path1 + "testSamples-360-withColour.npy")

for i in range(3):

    model = pd.read_pickle(os.getcwd() + "/../../model-pickle-files/" + svm_models[i] +"-SVM-model-360.pkl")
    #plot_roc_curve(model, testX, testY)
    predY = model.predict_proba(testX)

    #print(classification_report(testY, predY))
    pred_y = 0.5 + predY[:,1] - predY[:,0]

    rc = roc_curve(testY, pred_y)
    auc = int(roc_auc_score(testY, pred_y)*100)/100
    plt.plot(rc[0], rc[1], c=colours[3+i], label=labels[3+i] + str(auc))

plt.grid(True)
plt.legend(loc ="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('A graph to show the ROC curves of our models')
plt.show()