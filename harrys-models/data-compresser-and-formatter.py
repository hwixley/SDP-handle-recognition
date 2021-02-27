import os
import numpy as np
import pandas
from PIL import Image
import sklearn
import matplotlib.pyplot as plt

width = 1024
height = 576
numSamples = 1000
trainSamples = int(0.7 * numSamples)
testSamples = numSamples - trainSamples


def processImage(image):
    imArray = np.asarray(image.resize((width, height))).reshape(1, -1)
    return imArray


##Load images
trainX = np.empty((trainSamples, width * height * 3))

testX = np.empty((testSamples, width * height * 3))

for trX in range(trainSamples):
    sample = Image.open(os.getcwd() + "/../dataset/train/" + str(trX) + ".jpg")
    trainX[trX, :] = processImage(sample)

np.save(os.getcwd() + "/../../data/npy-data/trainSamples-" + str(height) + "-withColour.npy", trainX)

for teX in range(testSamples):
    sample = Image.open(os.getcwd() + "/../dataset/test/" + str(teX) + ".jpg")
    testX[teX, :] = processImage(sample)

np.save(os.getcwd() + "/../../data/npy-data/testSamples-" + str(height) + "-withColour.npy", testX)

