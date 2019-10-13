from mlp import MLPClassifier
from arff import Arff
import numpy as np
import pandas as pd


mat = Arff("../datasets/linsep2nonorigin.arff")
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
BClass = MLPClassifier(LR=0.1,momentum=0.5,shuffle=False,deterministic=10)
BClass.fit(data,labels)


# mat = Arff("../datasets/data_banknote_authentication.arff")
# np_mat = mat.data
# data = np_mat[:,:-1]
# labels = np_mat[:,-1].reshape(-1,1)
#
# #### Make Classifier and Train####
# B2Class = BPClassifier(LR=0.1,momentum=0.5,shuffle=False,deterministic=10)
# B2Class.fit(data,labels)
