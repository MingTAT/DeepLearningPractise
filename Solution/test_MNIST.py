#!/usr/bin/python

import matplotlib.pyplot as plt
from Data import *
import numpy as np
from NeuralNetwork import *

def main():
  mnist=Data.fromName('MNIST')
  mnist.normalize()
  mnist.toOneHot()

  np.random.seed(1)

  M=1E-1  # initial weight size
  nInputs=784

  layers=[(10, Linear)]
  title = 'Single layer, 10 neurons'

  #layers=[(50, ReLU), (50, ReLU), (10, Linear)]
  #title = 'Three layers, [(50, ReLU), (50, ReLU), (10, Linear)]'

  #layers = [(50, ReLU), (10, Linear), (50, ReLU), (10, Linear)]
  #title = 'Four layers, [(50, ReLU), (10, Linear), (50, ReLU), (10, Linear)]'

  CE=ObjectiveFunction('crossEntropyLogit')
  nn=NeuralNetwork(nInputs, layers, M)

  nIter=10000
  B=100
  eta=0.1 # learning rate

  train_errors = list()
  test_errors = list()

  for i in range(nIter):
    x, y = mnist.next_batch(B)

    logit = nn.doForward(x)
    J=CE.doForward(logit, y)
    dp = CE.doBackward(y)
    # DEBUG

    nn.doBackward(dp)
    nn.updateWeights(eta)

    if (i%200==0):
      # training error
      p = nn.doForward(x)
      yhat = p.argmax(axis=0)
      yTrue = y.argmax(axis=0)
      accu = 1 - sum(yhat == yTrue) / len(yTrue)
      train_errors.append(accu)

      # testing error
      p=nn.doForward(mnist.x_test)
      yhat=p.argmax(axis=0)
      yTrue=mnist.y_test.argmax(axis=0)
      accu = 1 - sum(yhat==yTrue)/len(yTrue)
      #print( '\riter %d, J=%f, accu=%.2f' % (i, J, accu))
      test_errors.append(accu)

  plt.plot([i * 200 for i in range(len(train_errors))], train_errors, label = 'training error')
  plt.plot([i * 200 for i in range(len(train_errors))], test_errors, label = 'testing error')
  plt.legend()
  plt.title(title)
  plt.show()

main()