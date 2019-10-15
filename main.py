from mlp import MLPClassifier
from arff import Arff
import numpy as np
import pandas as pd


mat = Arff("datasets/linsep2nonorigin.arff")

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

# FIT ON TRAIN DATA

num_features = np.size(data, 1)
num_inputs = np.size(data, 0)
num_outputs = 1
hiddenLayers = [num_features, num_features * 2, num_outputs]
print("LAYER INFO INPUT")
print(hiddenLayers)

MLP = MLPClassifier(hiddenLayers,lr=0.1,momentum=0.5,shuffle=False, validationSize=0.0, deterministic=10)
trainData, trainLabels, testData, testLabels = MLP.splitTestTrainData(data, labels)
MLP.fit(trainData, trainLabels)
print("SCORE", MLP.score(trainData, trainLabels))

print("TESTSCORE", MLP.score(testData, testLabels))


# mat = Arff("../datasets/data_banknote_authentication.arff")
# np_mat = mat.data
# data = np_mat[:,:-1]
# labels = np_mat[:,-1].reshape(-1,1)
#
# #### Make Classifier and Train####
# B2Class = BPClassifier(LR=0.1,momentum=0.5,shuffle=False,deterministic=10)
# B2Class.fit(data,labels)
