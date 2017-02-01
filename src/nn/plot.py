# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:14:04 2016

3.4 Plots first layer of weights 
@author: risal
"""

from util import Load, Save, DisplayPlot
import matplotlib.pylab as plt
import numpy as np
from nn import Softmax

fname_1 = 'nn_model_confidence.npz'
fname_2 = 'cnn_model_batch_500.npz'
nn_model = Load(fname_1)
cnn_model = Load(fname_2)

#DisplayPlot(stats['train_ce'], stats['valid_ce'], 'Cross Entropy', number=0)
#DisplayPlot(stats['train_acc'], stats['valid_acc'], 'Accuracy', number=1)

def ShowWeightsNN(weights, number=0):
  """Show the weights centers as images."""
  plt.figure(number)
  plt.clf()
  for i in xrange(weights.shape[1]):
    plt.subplot(1, weights.shape[1], i+1)
    plt.imshow(weights[:, i].reshape(48, 48), cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')

def ShowWeightsCNN(weights, number=0):
  """Show the weights centers as images."""
  plt.figure(number)
  plt.clf()
  for i in xrange(weights.shape[1]):
    plt.subplot(1, weights.shape[1], i+1)
    plt.imshow(weights[:, i].reshape(5, 5), cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')
  
  
ShowWeightsNN(nn_model['W1'],1)

cnn_filters = np.sum(cnn_model['W1'],axis=2)


ShowWeightsCNN(np.reshape(cnn_filters,[25,8]),2)


