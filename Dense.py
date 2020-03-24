import numpy as np 

class Dense_Layer:
  def __init__(self, inputs, neurons):
    self.weights = np.random.randn(inputs, neurons) / np.sqrt(inputs + neurons)
    self.biases = np.zeros((1, neurons))
    self.momentum = False 
    self.weight_momentum = 0
    self.bias_momentum = 0 
    self.rmsprop = False
    self.eG2weights = 0
    self.eG2biases = 0 
    

  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
    self.inputs = inputs

  def backward(self, derivatives):
    self.dweights = np.dot(self.inputs.T, derivatives)
    self.dbiases = np.sum(derivatives, axis = 0)
    self.derivatives = np.dot(derivatives, self.weights.T)