import numpy as np
import math 
import time
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from matplotlib import style 
from sklearn.datasets import make_circles
from Dense import Dense_Layer
from Activation.ReLU import Activation_ReLU
from Activation.Softmax import Activation_Softmax
from CostFunctions.Cross_Entropy import Loss_CategoricalCrossEntropy
from Optimizers.sgd import SGD

# Function that generates nonlinear data
np.random.seed(0)

def create_data():
  X, y = make_circles(n_samples=1000, factor=.3, noise=.10)
  return X, y

def create_data2(points, classes):
  X = np.zeros((points*classes,2))
  Y = np.zeros(points*classes, dtype = 'uint8')
  for class_number in range(classes):
    ix = range(points*class_number, points*(class_number+1))
    t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
    r = np.linspace(0.0, 1, points)  # radius
    X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
    Y[ix] = class_number
  return X, Y

# X is an array of samples or coordinate pairs, Y is the label for where the data falls
def plot_train(X,y):
  plt.scatter(X[:,0],X[:,1], c = y)
  plt.show()

dense1 = Dense_Layer(2,64)
activation1 = Activation_ReLU()
dense2 = Dense_Layer(64,3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()
optimizer = SGD(learning_rate=1.0,decay=5e-8, momentum=0.90)
acc = []

X,y = create_data2(100,3)

style.use("fast")

def run_net(iterations, plot = True):
  if plot == True:
    data = []
    data2 = []
    data3 = []
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 6)
    ax1 = fig.add_subplot(gs[0, 4:])
    ax2 = fig.add_subplot(gs[1, 4:])
    ax3 = fig.add_subplot(gs[-1, 4:])
    line, = ax1.plot(data)
    line2, = ax2.plot(data2)
    line3, = ax3.plot(data3)
    plt.ion()
    plt.show()
  for i in range(iterations): 
    # Forward Prop
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_function.forward(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1) 
    accuracy = np.mean(predictions==y)

    if plot == True:
      data.append(accuracy)
      data2.append(loss)
      data3.append(optimizer.learning_rate)
      ax1.set_xlim([0,i])
      ax1.set_ylim([0.33, 1.0])
      ax2.set_ylim([min(data2), 1.1])
      ax2.set_xlim([0,i])
      ax3.set_xlim([0,i])
      ax3.set_ylim([min(data3), max(data3)])
      line.set_ydata(data)
      line.set_xdata(range(len(data)))
      line2.set_ydata(data2)
      line2.set_xdata(range(len(data2)))
      line3.set_ydata(data3)
      line3.set_xdata(range(len(data3)))
      plt.pause(0.01)


    if i % 5 == 0:
      if i % 100 == 0:
        print("ep:", i, "loss:", loss, "acc:", accuracy, "LR", optimizer.learning_rate)
      # Backprop
      loss_function.backward(activation2.output,y)
      activation2.backward(loss_function.derivatives)
      dense2.backward(activation2.derivatives)
      activation1.backward(dense2.derivatives)
      dense1.backward(activation1.derivatives)
      # Weight update
      optimizer.pre_update()
      optimizer.update(dense1)
      optimizer.update(dense2)
      optimizer.post_update()

run_net(10000)  

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.forward(activation2.output, y)
predictions = np.argmax(activation2.output, axis=1) 
accuracy = np.mean(predictions==y)   

  


# def softmax_grad(softmax):
#     # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
#     s = softmax.reshape(-1,1)
#     return np.diagflat(s) - np.dot(s, s.T)

# def backward_propagation(X, forward_values, true_y, sample_size, relu_class,dense2):
#   predic_less_hot = forward_values["A2"]
#   predic_less_hot[0][y[0]] -= 1
#   dl_da = predic_less_hot # 1 x 3
#   dC_dz = np.dot(dl_da, softmax_grad(A2)) # 1 x 3 
#   dC_da_l1 = np.sum(dC_dz.reshape(-1,1) * dense2.weights, axis=0, keepdims = True)
#   dC_dw = dC_dz.reshape(-1,1) * A1 # 3 x 3 
#   dz_db = dC_dz # 1 x 3
#   dC_dz1 = activation1.derivative(A1) * dC_da_l1
#   dC_dw1 = dC_dz1.reshape(-1,1) * X
#   dC_db1 = dC_dz1

#   backprop_output = { "dW2": dC_dw, "dB2": dC_db, "dB1": dC_db1, "dW1": dC_dw1 }
#   return backprop_output

# def run_net(X, y, learning_rate, iteration):
#   dense1 = Dense_Layer(2,3)
#   activation1 = Activation_ReLU()
#   dense2 = Dense_Layer(3,3)
#   activation2 = Activation_Softmax()
#   loss_function = Loss_CategoricalCrossEntropy()
#   true_y = true_distribution(y)


#   for i in range(iteration):
#     # Move forward through network
#     for i in X:
#       forward_values, probabilities = forward_propagation(X, dense1, activation1, dense2, activation2)
#       # Loss calculation
#       loss, acc = loss_and_accuracy(probabilities, y, loss_function)
#       # Propogate backwards 
#       backward_output = backward_propagation(X, forward_values, true_y, sample_size, activation1, dense2)

#       # print('this is iteration', i)
#       print('new set of weights found, iteration', i, 'loss:', loss, 'acc:', acc)
#       # Update nets weights and biases
#       dense1.weights -= learning_rate*backward_output["dW1"]
#       dense1.biases -= learning_rate*backward_output["dB1"]
#       dense2.weights -= learning_rate*backward_output["dW2"]
#       dense2.biases -= learning_rate*backward_output["dB2"]

# ''' Below is me just trying to get some visibility '''
# X,y = create_data2(100,3)



# sample_size = len(y)

# true_y = true_distribution(y)

# forward_values, probs = forward_propagation(X, dense1, activation1, dense2, activation2)
# loss, accuracy = loss_and_accuracy(probs,y,loss_function)
# backward_output = backward_propagation(X, forward_values, true_y, sample_size, activation1,dense2)

# grad = forward_values["A2"]
# grad[range(len(y)),y] -= 1
# dZ2 = grad/len(y)


# def softmax_grad(softmax):
#     # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
#     s = softmax.reshape(-1,1)
#     return np.diagflat(s) - np.dot(s, s.T)

# X = [0.4,0.3]
# y = [1]
# sample_size = 2
# # forward start
# dense1.forward(X)
# Z1 = dense1.output
# activation1.forward(Z1)
# A1 = activation1.output
# dense2.forward(A1)
# Z2 = dense2.output
# activation2.forward(Z2)
# A2 = activation2.output

# # backward start
# predic_less_hot = A2
# predic_less_hot[0][y[0]] -= 1
# dl_da = predic_less_hot # 1 x 3
# dC_dz = np.dot(dl_da, softmax_grad(A2)) # 1 x 3 
# dC_da_l1 = np.sum(dC_dz.reshape(-1,1) * dense2.weights, axis=0, keepdims = True)
# dC_dw = dC_dz.reshape(-1,1) * A1 # 3 x 3 
# dz_db = dC_dz # 1 x 3
# dC_dz1 = activation1.derivative(A1) * dC_da_l1
# dC_dw1 = dC_dz1.reshape(-1,1) * X
# dC_db1 = dC_dz1
# print(dC_dw1)
# print(dC_db1)


# grad = forward_values["A2"]
# grad[range(len(y)),y] -= 1
# dZ2 = grad/len(y)
# dW2 = (1/sample_size) * np.dot(forward_values["A1"].T, dZ2) 
# dB2 = (1/sample_size) * np.sum(dZ2,axis=0,keepdims=True)
# dZ1 =  np.dot(dense2.weights,relu_class.derivative(dZ2).T)
# dW1 = (1 / sample_size) * np.dot(X.T, dZ1.T)
# dB1 = (1 / sample_size) * np.sum(dZ1,axis=1, keepdims=True).reshape(1,3)




# # dz_dw = ds_dz * A1 #### ?? but obviously with dot product 3 x 2 ? 
# # dz_db = ds_dz * 1 ### ?? Really 


# # m = 2
# # dW2 = (1 / m) * np.dot(dZ2, A1.T)
# # db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)


# # ====================================

# # Microsoft 

# soft_max = A2  



