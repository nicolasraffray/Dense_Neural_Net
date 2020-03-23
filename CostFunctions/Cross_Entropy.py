import numpy as np

class Loss_CategoricalCrossEntropy():

  def forward(self, y_pred, y_true):
    # Number of samples in a batch
    samples = y_pred.shape[0]
    # Probabilities for target values - only if categorical labels
    if len(y_true.shape) == 1:
        y_pred = y_pred[range(samples), y_true]
    # Losses
    negative_log_likelihoods = -np.log(y_pred)
    # Mask values - only for one-hot encoded labels
    if len(y_true.shape) == 2:
        negative_log_likelihoods *= y_true
    # Overall loss
    data_loss = np.sum(negative_log_likelihoods) / samples
    return data_loss

  def backward(self, derivatives, y_true):
      samples = derivatives.shape[0]
      self.derivatives = derivatives.copy()
      self.derivatives[range(samples), y_true] -= 1
      self.derivatives = self.derivatives / samples
