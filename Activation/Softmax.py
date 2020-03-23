import numpy as np

class Activation_Softmax():
  
  def forward(self, inputs):
    self.inputs = inputs
    # get unnormalized probabilities
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    # normalize them for each sample
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.output = probabilities


  def backward(self, derivatives):
    self.derivatives = derivatives.copy()