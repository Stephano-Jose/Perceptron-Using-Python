
import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, epoch=100, learning_rate=0.01):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0.0:
            activation = 1
        elif summation == 0.0:
            activation = 0            
        else:
            activation = -1
        return activation

    def train(self, training_inputs, labels):
        for i in range(self.epoch):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                if prediction!=label:
                    self.weights[1:] += self.learning_rate * (label-prediction) * inputs
                    self.weights[0] += self.learning_rate * (label-prediction)

