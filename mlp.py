import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import math

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=1, shuffle=True, validationSize=0.0, deterministic=False):
        """ Initialize class with chosen hyperparameters.
        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        #self.num_hidden_layers = len(hidden_layer_widths)
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.weights = []
        self.accuracy = 0
        self.validationSize = validationSize
        self.deterministic = deterministic
        self.Z1 = []

        self.dW1 = 0
        self.dW2 = 0

    def keepGoing(self, det, noChange):
        if(noChange >= 20): return False
        if(det <= 0): return False
        return True

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        #set training and validation data
        validData = []
        validLabels = []
        if(self.validationSize):
            trainData, trainLabels, validData, validLabels = self.splitValidation(X, y, self.validationSize)
        else:
            trainData = X
            trainLabels = y
        
        self.weights = self.initialize_weights() if not initial_weights else initial_weights
        print(self.weights)

        noChange = 0
        lastScore = 0
        det = self.deterministic
        #for i in range(0, det):#det):
        while(self.keepGoing(det, noChange)):
            #shuffle each epoch
            if(self.shuffle): trainData, trainLabels = self._shuffle_data(trainData, trainLabels)
            #stochastic! So iterate through the data and update as we go
            for row, label in zip(trainData, trainLabels): 
                row = np.append(row, 1)
                O2 = self.forward(row) #concat the 1 for bias
                loss = self.mse(O2, label)
                self.backprop(row, label)
            
            #stopping criteria
            if(det): det -= 1
            else:
                score = self.score(validData, validLabels)
                if(lastScore - score > .005):
                    noChange += 1
                lastScore = score
                if(noChange > 20): break
            
        return self

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def forward(self, Xrow):
        # print("Z1", self.Z1)
        # print("W2", self.weights['W2'])
        # print("XROWT", Xrow.T)
        self.Z1 = np.dot(self.weights['W1'], Xrow.T)
        self.Z1 = np.append(self.Z1, 1) #append bias to layer 2
        self.O1 = self.sigmoid(self.Z1)
        
        self.Z2 = np.dot(self.weights['W2'], self.Z1)
        self.O2 = self.sigmoid(self.Z2)

        return self.O2

    def backprop(self, row, labels):
        row = row.reshape(-1,1)

        O2 = self.O2.reshape(-1,1)#[0]

        S = (labels - O2)*(O2)*(1 - O2)
        dW2 = (self.lr * S * self.O1) + (self.momentum * (self.dW2))  #zhe shi yi ge array, suo yi treat as such

        S2 = self.O1 * (1-self.O1) * np.dot(S.T, self.weights['W2']) #* S * self.weights['W2']
        dW1 = (self.lr * row * S2[:,:S2.shape[1] - 1]) +  (self.momentum * (self.dW1))#Don't need to propogate bias backwards
        
        self.weights['W2'] += dW2
        self.weights['W1'] += dW1.T

        self.dW2 = dW2
        self.dW1 = dW1

        return

    def mse(self, output, label):
        squared_errors = (output - label) ** 2
        loss= np.sum(squared_errors)

        return loss

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        pred = self.forward(np.append(X, 1))
        pred = np.argmax(pred)
        print("--PRED--", pred)
        #ARGMAX???

        return pred

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        weights = {}
        np.random.seed(1)
        #Remember + 1 is for bias
        weights['W1'] = np.random.normal(0, 1, size=[self.hidden_layer_widths[1], self.hidden_layer_widths[0] + 1])
        weights['W2'] = np.random.normal(0, 1, size=[self.hidden_layer_widths[2], self.hidden_layer_widths[1] + 1])
        return  weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        correctCount = 0
        totalCount = 0

        for inputs, exp in zip(X, y):
            pred = self.predict(inputs)
            exp = np.asscalar(exp)
            print("pred", pred)
            print("exp", exp)
            if(pred == exp): correctCount += 1
            totalCount += 1

        self.accuracy = correctCount/totalCount
        #find num outputs that we got right
        return self.accuracy

    def splitTestTrainData(self, X, y):
        #75/25 for nuw
        if(self.shuffle): self._shuffle_data(X, y)

        # if(self.entireSet):
        #     self.trainData = self.allData
        #     self.testData = self.allData
        # else:
        numRows = np.size(X, 0)
        trainRows = math.floor(numRows / 10 * 75)

        trainData = X[0:trainRows, :]
        trainLabels = y[0:trainRows, :]

        testData = X[trainRows:, :]
        testLabels = y[trainRows:, :]
        return trainData, trainLabels, testData, testLabels
        
    def splitValidation(self, X, y, validSize):
        numRows = np.size(X, 0)
        if(self.shuffle): self._shuffle_data(X, y)

        trainRows = math.floor(numRows / validSize)

        trainData = X[0:trainRows, :]
        trainLabels = y[0:trainRows, :]

        validData = X[trainRows:, :]
        validLabels = y[trainRows:, :]
        return trainData, trainLabels, validData, validLabels

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        allData = np.concatenate((X, y), axis=1)

        np.random.shuffle(allData)
        
        y = allData[: , allData.shape[1]-1:]
        X = allData[:, :allData.shape[1]-1]
        return X, y

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
