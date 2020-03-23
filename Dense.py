import numpy as np 

class Layer_Dense:
  def __init__(self, inputs, neurons):
    self.weights = np.random.randn(inputs, neurons) / np.sqrt(inputs + neurons)
    self.biases = np.zeros((1, neurons))

  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
    self.inputs = inputs

  def backward(self, derivatives):
    self.dweights = np.dot(self.inputs.T, derivatives)
    self.dbiases = np.sum(derivatives, axis = 0)
    self.derivatives = np.dot(derivatives, self.weights.T)