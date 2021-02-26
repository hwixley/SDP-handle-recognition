import os
from PIL import Image
import numpy as np

path = os.getcwd() + "/../Synthetic-Data-Generator/imageai/"
#trainLabels = np.load(path + "old/training/trainLabels.npy")
#testLabels = np.load(path + "old/testing/testLabels.npy")

from imageai.Classification.Custom import ClassificationModelTrainer


model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory(path)
model_trainer.trainModel(num_objects=2, num_experiments=3, enhance_data=True, batch_size=32, show_network_summary=True)