import numpy as np
import pprint as pp
class nn():
    def __init__(self, input = 2, hidden = 10, output = 1):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        self.input_dim = input
        self.hidden_dim = hidden
        self.output_dim = output
        np.random.seed(0)
        self.W1 = np.random.randn(input + 1, hidden)
        self.W2 = np.random.randn(hidden + 1, output)

    def feedforward(self, X):
        self.X0 = np.insert(X, 0, 1, axis = 1)
        #pp.pprint(self.X0)
        self.S1 = self.X0.dot(self.W1)
        #pp.pprint(self.S1)
        self.X1_ = np.tanh(self.S1)
        self.X1 = np.insert(self.X1_, 0, 1, axis = 1)
        #pp.pprint(self.X1)
        self.S2 = self.X1.dot(self.W2)
        #print(np.shape(self.S2))
        #self.h = np.tanh(self.S2)

    def backpropagation(self, Y):
        self.d2 = 2 * (self.S2 - Y) #* (1 - self.h ** 2)
        self.dW2 = (self.X1.T).dot(self.d2)
        self.d1 = self.d2.dot(self.W2[1:].T) * (1 - np.power(self.X1_, 2))
        self.dW1 = (self.X0.T).dot(self.d1)

    def train(self, X, Y, epsilon = 0.00001):
        self.feedforward(X)
        self.backpropagation(Y)
        self.W1 += -epsilon * self.dW1
        self.W2 += -epsilon * self.dW2

    def perror(self, X, Y):
        self.feedforward(X)
        output = np.sign(self.S2)
        error = np.linalg.norm((output - Y), ord = 2)
        #error = len(Y) - np.sum(output == Y)
        #print("error: ", error)
        return error/len(Y)

    def predict(self, X):
        self.feedforward(X)
        return np.sign(self.S2)
