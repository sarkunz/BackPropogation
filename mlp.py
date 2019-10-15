import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True):
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
        self.initial_weights = []
        self.ch = {}


    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        if(self.shuffle): X, y = self._shuffle_data(X, y)
        self.num_tr_sams = y.shape[0]
        
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        print(self.initial_weights)

        for i in range(0, 3000):
            init_output = self.forward(X)
            loss = self.nloss(init_output, y)
            self.backprop(X, y)

            if(i % 500 == 0):
                print ("Cost after iteration %i: %f" %(i, loss))

        print("final loss ", loss)

        # init_output = self.forward(X)
        # print(init_output)
        # loss = self.nloss(init_output, y)
        # print(loss)
        # fin_output = self.backprop(X, y)

        return self

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def dSigmoid(self, Z):
        s = 1/(1+np.exp(-Z))
        dZ = s * (1-s)
        return dZ

    def forward(self, X):
        X = X.transpose()

        #activation
        self.Z1 = self.initial_weights['W1'].dot(X)#+ self.initial_weights['B1']
        np.append(self.Z1, self.initial_weights['B1'])
        #squishify
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = self.initial_weights['W2'].dot(self.A1)# + self.initial_weights['B2']
        np.append(self.Z2, self.initial_weights['B2'])
        self.init_outp = self.sigmoid(self.Z2)

        return self.init_outp

    def backprop(self, X, Y):
        X = X.transpose()
        Y = Y.transpose()

        dLoss_init_outp = -(np.divide(Y, self.init_outp) - np.divide(1 - Y, 1 - self.init_outp))

        dLoss_Z2 = dLoss_init_outp * self.dSigmoid(self.Z2) 
        dLoss_A1 = np.dot(self.initial_weights["W2"].T, dLoss_Z2)
        dLoss_W2 = 1./self.A1.shape[1] * np.dot(dLoss_Z2,self.A1.T)
        dLoss_b2 = 1./self.A1.shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 

        dLoss_Z1 = dLoss_A1 * self.dSigmoid(self.Z1)        
        dLoss_A0 = np.dot(self.initial_weights["W1"].T, dLoss_Z1)
        dLoss_W1 = 1./X.shape[1] * np.dot(dLoss_Z1, X.T)
        dLoss_b1 = 1./X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1])) 

        self.initial_weights["W1"] = self.initial_weights["W1"] - self.lr * dLoss_W1
        self.initial_weights["b1"] = self.initial_weights["B1"] - self.lr * dLoss_b1
        self.initial_weights["W2"] = self.initial_weights["W2"] - self.lr * dLoss_W2
        self.initial_weights["b2"] = self.initial_weights["B2"] - self.lr * dLoss_b2

        return

    def nloss(self, output, Y):
        Y = Y.transpose()
        loss = (1/self.num_tr_sams) * (-np.dot(Y,np.log(output).T) - np.dot(1-Y, np.log(1-output).T))
        return loss

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        pass

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        weights = {}
        np.random.seed(1)
        weights['W1'] = np.random.normal(0, 1, size=[self.hidden_layer_widths[1], self.hidden_layer_widths[0]])
        weights['B1'] = np.ones(self.hidden_layer_widths[1])
        weights['W2'] = np.random.normal(0, 1, size=[self.hidden_layer_widths[2], self.hidden_layer_widths[1]])
        weights['B2'] = np.ones(self.hidden_layer_widths[2])
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

        return 0

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
        return self.initial_weights
