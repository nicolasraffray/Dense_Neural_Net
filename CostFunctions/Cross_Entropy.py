import numpy as np

class Loss_CategoricalCrossEntropy():

  def forward(self, y_pred, y_true):
    samples = y_pred.shape[0]
    y = y_true 
    y_pred = y_pred[range(samples), y_true]
    negative_log_likelihoods = -np.log(y_pred)
    data_loss = np.sum(negative_log_likelihoods) / samples
    return data_loss

  def backward(self, derivatives, y_true):
      samples = derivatives.shape[0]
      self.derivatives = derivatives.copy()
      self.derivatives[range(samples), y_true] -= 1
      self.derivatives = self.derivatives / samples
