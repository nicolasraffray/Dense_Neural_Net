import numpy as np

class Activation_ReLU:

  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0.0,inputs)

  def backward(self, dZ):
    dZ[self.inputs <= 0] = 0 
    self.derivatives = dZ 