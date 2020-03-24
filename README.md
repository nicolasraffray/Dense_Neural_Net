## Current Status 

I've deaded out and deleted a lot of code form inside the for loop of the run_nn function. Was just trying to get a hang of the backprop keeping it there right now for reference

Also this series is amazing highly reccomend 
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

Also sorry for the long ass readme. 

## DATA 

Generating Datasets: https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/

So the data is just some function for plotting non-linear data I found online. There are other functions you can try out if you look for them on medium too. Definitely take a look before jumping into the code and play around with changing no of "classes or y variable". 

run in terminal python -i simple_net.py

``` python 
X, y = create_data2(100,3)
plot_train(X,y)
```

Note rn I'm net is training on 100% of the dataset. What i think we should do next if this fits the set is use the first create_data function and split it into training set and test set and see how it does with different ratios of training data against test data

## Network

Purpose: Its trying to classify points on a graph correctly

I used ReLU just bc its easy and commonly used but could try with sigmoid which used to and still is quite popular. 

Categorical cross entropy loss is, I think, the most used cost function for classification w/ softmax. 

I found this really helpful: https://datascience.stackexchange.com/questions/20296/cross-entropy-loss-explanation

where I found the derivative :p :  https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function

From what I've read its just upstream grad * local grad where upstream grad in first step of backprop is dL/dL = 1. 

This resource is great: https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html
 
## Running net 

If you fork the repo and in the terminal run 
$ python3 -i simple_net.py

NB: To stop the net running close the matplotlib window that pops up and then ctrl + C in your terminal

```python
run_net(10000)
```

This will run the net through 10000 iterations and some dynamic plotting of the learning rate, accuracy and loss. Note with the optimization currently implemented it takes until around the 3000/4000th iteration before the accuracy really starts to climb.


