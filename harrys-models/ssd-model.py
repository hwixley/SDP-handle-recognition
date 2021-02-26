import os
import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from PIL import Image
from arcgis.learn import SingleShotDetector
import torch

path = os.getcwd() + "/../dataset/"
res = "360"

trainX = np.load(path + "training/trainSamples"+res+".npy")
trainY = np.load(path + "training/trainLabels.npy").astype("int")
testX = np.load(path + "testing/testSamples"+res+".npy")
testY = np.load(path + "testing/testLabels.npy").astype("int")

ssd = SingleShotDetector(trainX, grids=[4], zooms=[1.0], ratios=[[1.0, 1.0],[10.0,20.0]])