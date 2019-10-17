from mlp import MLPClassifier
from arff import Arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def oneHotCode(labels): #to rule them all
    onehotencoder = OneHotEncoder(categorical_features = [0])
    hottie = onehotencoder.fit_transform(labels).toarray()
    return hottie


#debug dataset 1

print("----------DEBUG DATASET 1------")
mat = Arff("datasets/linsep2nonorigin.arff")
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
num_features = np.size(data, 1)
num_inputs = np.size(data, 0)
num_outputs = 2
hiddenLayers = [num_features, num_features * 2, num_outputs]
print("LAYER INFO INPUT")
print(hiddenLayers)

MLP = MLPClassifier(hiddenLayers,lr=0.1,momentum=0.5,shuffle=False, validationSize=0.0, deterministic=10)
trainData, trainLabels, testData, testLabels = MLP.splitTestTrainData(data, labels)

print("labels", trainLabels)
print("testlabels", testLabels)

weights = {}
weights['W1'] = np.zeros([hiddenLayers[1], hiddenLayers[0] + 1])
weights['W2'] = np.zeros([hiddenLayers[2], hiddenLayers[1] + 1])

MLP.fit(trainData, trainLabels, weights)
print(MLP.get_weights())

print("SCORE", MLP.score(testData, testLabels))


#HW PROB

#print("----------HW EX------")
# data = np.array([[0,0],[0,1]])
# labels = np.array([1,0])
# hiddenLayers = [2,2,2]

# MLP = MLPClassifier(hiddenLayers,lr=1,momentum=0.5,shuffle=False, validationSize=0.0, deterministic=1)
# weights = {}
# weights['W1'] = np.ones([hiddenLayers[1], hiddenLayers[0] + 1])
# weights['W2'] = np.ones([hiddenLayers[2], hiddenLayers[1] + 1])
# MLP.fit(data, labels, weights)
# print("SCORE", MLP.score(data, labels))


###DEBUG DATASET 2
#get data

print("----------DEBUG DATASET 2 (banknote)------")
# #mat = Arff("datasets/data_banknote_authentication.arff")
# mat = Arff("datasets/iris.arff")
# np_mat = mat.data
# data = np_mat[:,:-1]
# labels = np_mat[:,-1].reshape(-1,1)

# #set dimensions
# num_features = np.size(data, 1)
# num_inputs = np.size(data, 0)
# num_outputs = np.size(np.unique(labels, 1),1)
# hiddenLayers = [num_features, num_features * 2, num_outputs]
# print("LAYER INFO INPUT")
# print(hiddenLayers)

# #MLP = MLPClassifier(hiddenLayers,lr=0.1,momentum=0.5,shuffle=True, validationSize=0.0, deterministic=10)
# MLP = MLPClassifier(hiddenLayers,lr=0.1,momentum=0.5,shuffle=True, validationSize=0.25, deterministic=False)
# trainData, trainLabels, testData, testLabels = MLP.splitTestTrainData(data, labels)


# MLP.fit(trainData, trainLabels)
# print(MLP.get_weights())
# print("SCORE", MLP.score(testData, (testLabels)))
