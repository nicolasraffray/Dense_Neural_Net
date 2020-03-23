import numpy as np

class SGD:
  def __init__(self, learning_rate = 0.1, decay = 0):
    self.learning_rate = learning_rate
    self.decay_rate = decay
    self.iterations = 0 

  def pre_update(self):
    self.learning_rate = self.learning_rate / (1 + self.decay_rate)

  def update(self, layer):
    layer.weights -= self.learning_rate * layer.dweights
    layer.biases  -= self.learning_rate * layer.dbiases

  def post_update(self):
    self.iterations += 1 

