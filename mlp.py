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
        self.num_tr_sams = y.shape[0]
        
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        print(self.initial_weights)

        output = self.forward(X)
        print(output)
        loss = self.nloss(output, y)
        print(loss)

        return self

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def dSigmoid(self, Z):
        s = 1/(1+np.exp(-Z))
        dZ = s * (1-s)
        return dZ

    def forward(self, X):
        print("FORWARD-------")
        X = X.transpose()

        Z1 = self.initial_weights['W1'].dot(X)#+ self.initial_weights['B1']
        np.append(Z1, self.initial_weights['B1'])
        A1 = self.sigmoid(Z1)

        Z2 = self.initial_weights['W2'].dot(A1)# + self.initial_weights['B2']
        np.append(Z2, self.initial_weights['B2'])
        A2 = self.sigmoid(Z2)

        output = A2
        print("OUT")
        print(output)
        return output

    def backprop(self, Y):
        Y = Y.transpose()
        #dloss = (1/self.num_tr_sams) * (-np.dot(Y,np.log(output).T) - np.dot(1-Y, np.log(1-output).T))

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
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        pass
