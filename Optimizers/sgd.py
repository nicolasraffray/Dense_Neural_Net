import numpy as np

class SGD:
  def __init__(self, learning_rate = 0.1, decay = 0, momentum = 0, rmsprop = 0):
    self.learning_rate = learning_rate
    self.decay_rate = decay
    self.iterations = 0 
    self.momentum = momentum
    self.rmsprop = rmsprop

  def pre_update(self):
    if self.decay_rate:
      self.learning_rate = self.learning_rate / (1 + self.decay_rate * self.iterations)

  def update(self, layer):
    if self.momentum: 
      if not layer.momentum:
        layer.weight_momentum = np.zeros_like(layer.weights)
        layer.bias_momentum = np.zeros_like(layer.biases)
        layer.momentum = True 

      layer.weight_momentum = self.momentum * layer.weight_momentum + (1-self.momentum)*layer.dweights
      layer.bias_momentum = self.momentum * layer.bias_momentum + (1-self.momentum)*layer.dbiases

    if self.rmsprop: 
      if not layer.rmsprop:
        layer.eG2weights = np.zeros_like(layer.weights)
        layer.eG2biases = np.zeros_like(layer.biases)
        layer.rmsprop = True
 
      layer.eG2weigths = self.rmsprop * layer.eG2weights + (1-self.rmsprop)*(layer.dweights ** 2) 
      layer.eG2biases = self.rmsprop * layer.eG2biases + (1-self.rmsprop)*(layer.dbiases ** 2)

    if self.rmsprop and self.momentum: 
      layer.weights -= (self.learning_rate/np.sqrt(layer.eG2weights)) * layer.weight_momentum
      layer.biases  -= (self.learning_rate/np.sqrt(layer.eG2biases)) * layer.bias_momentum
    elif self.momentum:
      layer.weights -= self.learning_rate * layer.weight_momentum
      layer.biases -= self.learning_rate * layer.bias_momentum
    elif self.rmsprop:
      layer.weights -= (self.learning_rate/np.sqrt(layer.eG2weights)) * layer.dweights
      layer.biases  -= (self.learning_rate/np.sqrt(layer.eG2biases)) * layer.dbiases
    else:
      layer.weights -= self.learning_rate * layer.dweights
      layer.biases  -= self.learning_rate * layer.dbiases

  def post_update(self):
    self.iterations += 1 

